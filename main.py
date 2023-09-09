from tqdm import tqdm
from functools import partial

import os
import sys
import wandb
from typing import List
from torch.utils.data import DataLoader

from torch import nn
from typing import Dict, Optional
import torch
from torch.optim import Optimizer, lr_scheduler
from torch.nn import Module
from typing import Callable

from flexibility_nn.utils import get_args, load_checkpoint, setup_logging, write_metrics_to_csv, count_parameters, \
    calculate_metrics
from flexibility_nn.datasets.datasets import load_data, get_transforms, PrefetchLoader, DataPrefetcher
from flexibility_nn.models.models import get_optimizer, get_model, CustomProjectionModel
from accelerate import Accelerator
from flexibility_nn.projectors import IDModule, CombinedRDKronFiLM
from flexibility_nn.hessian.hess_vec_prod import min_max_hessian_eigs, eff_dim

from flexibility_nn.utils import get_num_examples, handle_num_examples_and_batch_size, \
    load_checkpoint_if_required, prepare_wandb_if_required, prepare_data_loaders, prep_log_dir

# Type Definitions
Model = Module
Device = torch.device
Metrics = Dict[str, float]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def train_model(
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        accelerator: 'Accelerator',
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> Dict[str, float]:
    model.train()
    inputs, targets = inputs.to(device), targets.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    outputs = model(inputs)

    # compute loss
    loss = criterion(outputs, targets)

    # backward pass and optimization
    accelerator.backward(loss)

    optimizer.step()

    # Update learning rate
    lr_scheduler.step()

    total_loss = loss.item()
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()

    return {
        'train_loss': total_loss,
        'train_accuracy': correct / total,
        #'grad_norm': grad_norm,
    }


def main_training_loop(args, net: Model, trainloader_trim: DataLoader, device: Device, accelerator: Accelerator,
                       optimizer: Optimizer, lr_scheduler: lr_scheduler._LRScheduler, criterion: LossFunction,
                       testloader: DataLoader, testloader_org: DataLoader,
                       trainloader_trim_test: DataLoader, test_classes: Optional[List[int]], csv_file_path: str,
                       csv_headers: List[str], log_dir: str, num_examples: int):
    step_counter = total_loss = total_acc = counter = 0
    finished_run = False
    acc_train = 0.
    acc_test = 0.
    acc_test_org = 0.
    grad_train = 0.

    with tqdm(
            initial=0,
            total=args.total_steps,
            disable=not accelerator.is_main_process,
    ) as pbar:
        while step_counter < args.total_steps and not finished_run:

            data_iterator = iter(trainloader_trim)  # Create a new iterator for each epoch
            num_of_steps_curent_epoch = 0
            for inputs, targets in data_iterator:
                train_results = train_model(model=net, inputs=inputs, targets=targets, criterion=criterion,
                                            optimizer=optimizer, device=device, accelerator=accelerator,
                                            lr_scheduler=lr_scheduler)
                num_of_steps_curent_epoch+=1
                total_acc += train_results['train_accuracy']
                total_loss += train_results['train_loss']
                step_counter += 1
                counter += 1
                train_loss = total_loss / counter
                avg_acc = train_results['train_accuracy']
                #grad_norm += train_results['grad_norm']
                if step_counter % args.num_to_print == 0:
                    net.eval()
                    counter, total_acc, total_loss = 0, 0., 0.
                    func_metrics = partial(calculate_metrics, model=net, device=device, criterion=criterion)
                    acc_train, loss_train, grad_train = func_metrics(data_loader=iter(trainloader_trim_test), num_of_steps=200)
                    acc_test, loss_test ,grad_test= func_metrics(data_loader=iter(testloader), num_of_steps=200)
                    acc_test_org, loss_test_org ,grad_org= func_metrics(data_loader=iter(testloader_org), num_of_steps=200)

                    metrics = {
                        'step': step_counter,
                        'train_loss': loss_train,
                        'train_accuracy': avg_acc,
                        'test_loss': loss_test,
                        'train_accuracy_total': acc_train,
                        'test_accuracy_total': acc_test,
                        'test_accuracy_total_org': acc_test_org,
                        'test_loss_total_org': loss_test_org,
                        'grad_norm':grad_train,
                        'eff_dim_val': 0.
                    }
                    write_metrics_to_csv(csv_file_path, csv_headers, metrics)
                    if args.save_model:
                        torch.save(net, log_dir / 'model.ckpt')

                    net.train()

                if acc_train >= args.train_threshold:
                    finished_run = True
                    break
                if args.use_wandb:
                    accelerator.log({"Train Loss": train_loss, "Train Accuracy": avg_acc,
                                     "Test_loss": loss_test,
                                     'Test accuracy': acc_test}, step=step_counter)

                pbar.set_description(
                    f'Train Loss: {train_loss:.6f}, Train Accuracy: {avg_acc:.4f}, Train Accuracy Total: {acc_train:.4f}, '
                    f' Test Accuracy Total: {acc_test:.4f}, Test Acc Org: {acc_test_org:.4f}, Grad Norm: {grad_train}')
                pbar.update()

        print(f'finish - num_examples: {num_examples}, finished_run: {finished_run}')
        if not args.no_hessian_eigs:
            max_eval, min_eval, hvps, pos_evals, neg_evals = min_max_hessian_eigs(
                net, trainloader_trim, criterion, use_cuda=True, verbose=True
            )
            eff_dim_val = eff_dim(pos_evals.cpu().numpy())
            #eigenvals, eigenvecs = compute_hessian_eigenthings(net, trainloader_trim,
            #                                                   criterion, 10)


        else:
            eff_dim_val = -1

        metrics['eff_dim_val'] = eff_dim_val
        write_metrics_to_csv(csv_file_path, csv_headers, metrics)
        if args.use_wandb:
            accelerator.end_training()
            wandb.finish()

    return finished_run


def initialize_logging(args, accelerator, log_dir, file_path):
    prepare_wandb_if_required(args, accelerator, log_dir)
    with open(file_path, "w") as f:
        f.write(str(args))


def get_training_components(args, net):
    func_optimizer = partial(get_optimizer, net=net, optimizer_choice=args.optimizer, lr=args.lr,
                             weight_decay=args.weight_decay, momentum=args.momentum,
                             total_steps=args.total_steps)
    optimizer, lr_scheduler, criterion = func_optimizer()
    return optimizer, lr_scheduler, criterion


def prepare_model_and_data(args, trainloader, num_classes, input_size, original_batch, num_examples):
    handle_num_examples_and_batch_size(args, trainloader, num_examples, original_batch)

    net = get_model(
        architecture=args.architecture,
        num_classes=num_classes,
        input_size=input_size,
        hidden_layers=args.hidden_layers,
        nonlinearity=args.nonlinearity,
        args=args)
    if args.projection_dim < 1.:
        N = count_parameters(net)
        M = int(args.projection_dim * N)
        net = IDModule(net, CombinedRDKronFiLM, M)
    total_parameters = count_parameters(net)
    args.total_parameters = total_parameters
    if args.full_batch:
        args.batch_size = args.num_examples
    trainloader_trim, trainloader_trim_test = prepare_data_loaders(args, trainloader)
    return net, trainloader_trim, trainloader_trim_test


def main():
    args: object = get_args()
    csv_headers = ['step', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy', 'grad_norm',
                   'train_accuracy_total', 'test_accuracy_total', 'test_accuracy_total_org', 'eff_dim_val',
                   'test_loss_total_org']

    num_examples_points = get_num_examples(args)
    transform_train, transform_test, input_size = get_transforms(dataset=args.dataset)

    trainloader, testloader, testloader_org, num_classes, test_classes = load_data(
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=args.batch_size,
        dataset=args.dataset,
        data_dir=args.data_dir,
        random_input=args.random_input,
        random_labels=args.random_labels,
        workers=args.workers,
        num_new_classes=args.num_new_classes
    )

    accelerator = Accelerator(log_with="wandb", project_dir=".")
    device = accelerator.device
    original_batch = args.batch_size
    original_log_dir = args.log_dir

    print (num_examples_points)
    for num_examples in num_examples_points:
        csv_file_path, file_path, log_dir = prep_log_dir(original_log_dir, args)

        args.num_examples = num_examples
        net, trainloader_trim, trainloader_trim_test = prepare_model_and_data(
            args=args, trainloader=trainloader, num_classes=num_classes, input_size=input_size,
            original_batch=original_batch, num_examples=num_examples)

        optimizer, lr_scheduler, criterion = get_training_components(args, net)

        net, optimizer, trainloader_trim, lr_scheduler, testloader, trainloader_trim_test = accelerator.prepare(
            net, optimizer, trainloader_trim, lr_scheduler, testloader, trainloader_trim_test
        )

        initialize_logging(args=args, accelerator=accelerator, log_dir=log_dir, file_path=file_path)
        print(args)
        finished_run = main_training_loop(
            args=args, net=net, trainloader_trim=trainloader_trim, device=device, accelerator=accelerator,
            optimizer=optimizer, lr_scheduler=lr_scheduler, criterion=criterion,
            testloader=testloader, testloader_org=testloader_org, trainloader_trim_test=trainloader_trim_test,
            test_classes=test_classes, csv_file_path=csv_file_path, csv_headers=csv_headers, log_dir=log_dir,
            num_examples=num_examples
        )
        #if not finished_run:
        #    break


if __name__ == "__main__":
    main()
