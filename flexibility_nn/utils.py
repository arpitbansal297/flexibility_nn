import argparse
import yaml
import torch
import logging
from pathlib import Path
import wandb
from typing import Tuple, Any, Union
import torch.nn as nn
import csv
from datetime import datetime
import os
from flexibility_nn.datasets.utils import DataLoaderWithPrefetch
from typing import List
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import shutil
from itertools import cycle
from torchvision.datasets import ImageFolder
import random

from typing import Dict, Optional
import torch
from torch.optim import Optimizer, lr_scheduler
from torch.nn import Module
from typing import Callable
from torch import Tensor
import pandas as pd
from datetime import datetime
import os
from pathlib import Path



def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10')
    parser.add_argument('--start_num_examples', type=int, default=4, help='Starting number of examples')
    parser.add_argument('--end_num_example', type=int, default=50_000, help='Ending number of examples')
    parser.add_argument('--num_runs', type=int, default=20, help='Number of points to generate')
    parser.add_argument('--num_new_classes', type=int, default=None, help='Number of classes')
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging (default: False)")
    parser.add_argument("--projection_dim", type=float, default=1., help="Train lower dimension")
    parser.add_argument("--train_threshold", type=float, default=99.5,
                        help="The accuracy threshold to stop the training ")
    parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer choice: sgd, adam, or adamw')
    parser.add_argument('--nonlinearity', default='relu', type=str,
                        help='Nonlinearity choice: relu, sigmoid, tanh, leakyrelu, or linear')
    parser.add_argument("--num_filters", type=int, default=32,
                        help="Number of filters in the first convolutional layer")
    parser.add_argument("--hidden_size", type=int, default=50, help="Size of the hidden fully connected layer")
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help="dataset to use (cifar10, imagenet, inaturalist or cinic10)")
    parser.add_argument("--data_dir", default="/scratch/rs8020/data", type=str, help="directory containing the dataset")
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint in log_dir')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs to train')
    parser.add_argument('--total_steps', type=int, default=400_000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128 * 5, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay for optimizer')
    parser.add_argument('--log_dir', type=str, default='/scratch/rs8020/flexibility_dnn', help='where to log')
    parser.add_argument('--file_name', type=str, default='parameters.txt', help='The name of file to log params')
    parser.add_argument("--num_examples", default=10000, type=int, help="number of training examples")
    parser.add_argument("--architecture", default="mlp", type=str,
                        help="network architecture (mlp, efficientnet_b0-6, resnet9, resnet18, resnet34, resnet50, resnet101, or resnet152)")

    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser.add_argument('--name', default='default', type=str,
                        help='The name of the experiment')

    # Dataset parameters
    # Keep this argument outside the dataset group because it is positional.
    parser.add_argument('--data-dir', metavar='DIR',
                        help='path to dataset (root dir)')
    parser.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='validation',
                        help='dataset validation split (default: validation)')
    parser.add_argument('--dataset-download', action='store_true', default=False,
                        help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')

    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')

    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    # Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--aug-repeats', type=float, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')

    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 4)')

    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
    parser.add_argument('--no-hessian_eigs', action='store_true', default=True)

    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--log_every_n_steps', type=int, default=1,
                        help='Random erase prob (default: 0.)')

    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--random_labels', action='store_true', help="Use random labels instead of original labels")
    parser.add_argument('--save_model', action='store_true', help="Use random labels instead of original labels")
    parser.add_argument('--random_input', action='store_true', help="Use random input instead of original input")
    parser.add_argument('--full_batch', action='store_true', help="Use random input instead of original input")

    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    parser.add_argument('--num_to_print', type=int, default=10000,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--worker-seeding', type=str, default='all',
                        help='worker seed mode (default: all)')

    parser.add_argument('--rank', type=int, default=0,
                        help='worker seed mode (default: all)')

    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--grad_norm_threshold', default=1e-6, type=float,
                        help='Gradient norm threshold for early stopping (default: 1e-4)')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[1000, 1000],
                        help='Hidden layer sizes (default: [128, 64])')

    parser.add_argument('--swin_dims', type=int, nargs='+', default=[16, 32],
                        help='Hidden layer sizes (default: [16, 32])')
    parser.add_argument('--swin_num_blocks_list', type=int, nargs='+', default=[2],
                        help='Hidden layer sizes (default: [2])')
    parser.add_argument('--swin_head_dim', type=int, default=16,
                        help='swin_head_dim (default: 16)')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of hidden layers')

    args, args_text = _parse_args(parser, config_parser)
    return args


def _parse_args(parser, config_parser):
    args, args_text = parser.parse_known_args()
    if args.config:
        with open(args.config, 'r') as f:
            config_args = config_parser(f)
            args = parser.parse_args(config_args + args_text)
    return args, args_text


def save_checkpoint(state: dict, log_dir: Union[str, Path], filename: str = 'checkpoint.pth.tar') -> None:
    """
    Save checkpoint.

    :param state: State dictionary to be saved
    :param log_dir: Directory for saving the checkpoint
    :param filename: Checkpoint file name
    """
    torch.save(state, log_dir / filename)


def count_parameters(model):
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total_parameters = sum(p.numel() for p in model.parameters())
    return total_parameters


def npz_to_jpeg(root, new_root, total_samples):
    if os.path.exists(new_root):
        shutil.rmtree(new_root)
    os.makedirs(new_root)
    # Global counter for total samples saved
    total_saved_samples = 0

    # List of directories
    dir_names = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]

    # List of npz files for each directory, initialized as None
    dir_npz_files = {dir_name: None for dir_name in dir_names}
    all_gone = False

    while total_saved_samples < total_samples and not all_gone:
        all_gone = True
        for dir_name in dir_names:
            if total_saved_samples >= total_samples:
                # Stop the saving process if the total saved samples has reached the limit
                break

            # Get the list of npz files for this directory, or update it if it's empty
            if dir_npz_files[dir_name] is None or not dir_npz_files[dir_name]:
                dir_path = os.path.join(root, dir_name)
                dir_npz_files[dir_name] = [file for file in os.listdir(dir_path) if file.endswith('.npz')]

            # If there are no npz files left in this directory, skip it
            if not dir_npz_files[dir_name]:
                continue
            all_gone = False
            # Get and remove the first npz file from the list
            npz_file = dir_npz_files[dir_name].pop(0)
            npz_path = os.path.join(root, dir_name, npz_file)
            with np.load(npz_path) as data:
                samples = data['samples']
                samples = np.transpose(samples, (0, 2, 3, 1))

                for i, sample in enumerate(samples):
                    if total_saved_samples >= total_samples:
                        # Stop the saving process if the total saved samples has reached the limit
                        break

                    # Create the new directory if it doesn't exist
                    new_dir_path = os.path.join(new_root, dir_name)
                    os.makedirs(new_dir_path, exist_ok=True)
                    image = Image.fromarray(sample.astype(np.uint8))
                    image_path = os.path.join(new_dir_path, f'{npz_file.split(".npz")[0]}_{i}.jpg')
                    image.save(image_path)

                    # Increment the global counter
                    total_saved_samples += 1


def load_checkpoint(log_dir: Union[str, Path], filename: str = 'checkpoint.pth.tar') -> Any:
    """
    Load checkpoint.

    :param log_dir: Directory containing the checkpoint
    :param filename: Checkpoint file name
    :return: Loaded checkpoint
    """
    return torch.load(log_dir / filename)


def setup_logging(log_dir: Union[str, Path]) -> Path:
    """
    Setup logging.

    :param log_dir: Directory for log files
    :return: Path object for the log directory
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "training.log",
        filemode="w",
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    return log_dir


def write_to_csv(file_path, headers, data):
    file_exists = Path(file_path).exists()
    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def write_metrics_to_csv(file_path, headers, metrics):
    file_exists = file_path.exists()
    with open(file_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def filter_and_get_max(csv_path, full_batch, dataset, random_labels, random_input, architecture,
                       eval_measure='train_accuracy_total', eval_measure_val=0., cond_name='train_accuracy_total',
                       hidden_layers=None):
    # 1. Load the CSV into a Pandas DataFrame
    df = pd.read_csv(csv_path)

    # 2. Filter based on the provided properties
    mask = (
            (df['full_batch'] == full_batch) &
            (df['dataset'] == dataset) &
            (df['random_labels'] == random_labels) &
            (df['random_input'] == random_input) &
            (df['architecture'] == architecture)
    )

    if architecture == 'mlp' and hidden_layers is not None:
        mask = mask & (df['hidden_layers'] == hidden_layers)
    filtered_df = df[mask]
    print('filtered_df ', filtered_df.shape)

    # 3. Further filter based on eval_measure and eval_measure_val
    print('filtered_df1 ', filtered_df.shape)

    # 4. Return the maximum value of the cond_name column
    return filtered_df[cond_name].max()


def get_num_examples(args) -> List[int]:
    if args.start_num_examples == -1:
        return [-1]
    elif args.start_num_examples == -2:
        max_cond_name = filter_and_get_max("./data.csv", full_batch=args.full_batch, dataset=args.dataset,
                                           random_labels=args.random_labels, random_input=args.random_input,
                                           architecture=args.architecture,
                                           eval_measure_val=0.98, eval_measure='train_accuracy_total',
                                           cond_name='total_parameters', hidden_layers=str(args.hidden_layers))
        args.start_num_examples = max_cond_name * (1.2) + 50
        print(max_cond_name, args.start_num_examples)
    return np.unique(
        np.logspace(np.log10(args.start_num_examples), np.log10(args.end_num_example), num=args.num_runs,
                    dtype=int))


def handle_num_examples_and_batch_size(args, trainloader: DataLoader, num_examples: int, original_batch: int):
    args.num_examples = num_examples
    if num_examples <= 0 or num_examples > len(trainloader.dataset):
        args.num_examples = len(trainloader.dataset)
    args.name = 'i_' + str(args.num_examples)
    args.batch_size = original_batch
    if args.batch_size <= 1:
        args.batch_size = 2
    if args.full_batch:
        args.batch_size = args.num_examples


def load_checkpoint_if_required(args, net, optimizer, lr_scheduler, start_epoch, timestamp):
    if args.resume:
        if args.checkpoint_path is None:
            print("Please provide a checkpoint path using --checkpoint_path")
            return

        print("=> Loading checkpoint")
        checkpoint = load_checkpoint(Path(args.checkpoint_path))
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        return


def prepare_wandb_if_required(args, accelerator, log_dir):
    if args.use_wandb:
        # run = wandb.init(project="cifar10", name=args.name, config=vars(args), dir=str(log_dir))
        accelerator.init_trackers("cifar10_2", config={},
                                  init_kwargs={
                                      "wandb": {"name": args.name, 'config': vars(args), 'dir': str(log_dir)}})


def copy_images(root_path, destination_path, total_samples, batch_size, random_dest=False):
    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)
    os.makedirs(destination_path)
    dirs = [os.path.join(root_path, dir_name) for dir_name in os.listdir(root_path) if
            os.path.isdir(os.path.join(root_path, dir_name))]
    dir_image_files = {
        dir_name: cycle([file for file in os.listdir(dir_name) if file.endswith('.jpg') or file.endswith('.jpeg')]) for
        dir_name in dirs}

    copied_samples = 0
    while copied_samples < total_samples:
        for dir_name, cycle_image_files in dir_image_files.items():
            for _ in range(batch_size):
                if copied_samples >= total_samples:
                    break

                # Create same directory structure in destination
                if random_dest:
                    # Select random destination directory
                    destination_dir = random.choice(dirs).replace(root_path, destination_path)
                else:
                    destination_dir = dir_name.replace(root_path, destination_path)

                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)

                image_file = next(cycle_image_files)
                shutil.copy(os.path.join(dir_name, image_file), os.path.join(destination_dir, image_file))
                copied_samples += 1


def prepare_data_loaders(args, trainloader, root='/state/partition1/rs8020/'):
    if args.num_examples is not None and args.num_examples > 0:
        if args.dataset == 'imagenet_sampled':
            root_path = root + '/data'
            destination_path = root + '/data1'
            copy_images(root_path, destination_path, args.num_examples, 512, random_dest=args.random_labels)
            train_data_subset = ImageFolder(root=trainloader.dataset.root, transform=trainloader.dataset.transform)
        else:
            perm = torch.randperm(len(trainloader.dataset))
            idx = perm[:args.num_examples]
            train_data_subset = torch.utils.data.Subset(trainloader.dataset, idx)
        trainloader_trim = DataLoaderWithPrefetch(train_data_subset, batch_size=int(args.batch_size), shuffle=True,
                                                  num_workers=args.workers, persistent_workers=True, pin_memory=True,
                                                  prefetch_size=args.workers * 1000)

        trainloader_trim_test = DataLoaderWithPrefetch(train_data_subset, batch_size=int(args.batch_size),
                                                       shuffle=False, num_workers=args.workers,
                                                       persistent_workers=True, pin_memory=True,
                                                       prefetch_size=args.workers * 1000)
    else:
        trainloader_trim = trainloader
        trainloader_trim_test = trainloader

    return trainloader_trim, trainloader_trim_test



def prep_log_dir(original_log_dir, args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version = 1
    max_version = 10  # Maximum version number

    # Loop to find a non-existing log directory
    while version <= max_version:
        log_dir_name = f"{original_log_dir}/logs/{timestamp}_v{version}"
        log_dir = Path(log_dir_name)

        if not log_dir.exists():
            break
        version += 1

    if version > max_version:
        print("Reached the maximum version number. Exiting.")
        exit(1)

    log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = log_dir

    file_path = os.path.join(log_dir, args.file_name)
    csv_file_path = log_dir / 'results.csv'

    return csv_file_path, file_path, log_dir


def get_grad_norm(model_params):
    norm = 0
    for p in model_params:
        try:
            norm += torch.linalg.norm(p.grad.detach().data).item()**2
        except:
            pass
    return norm**0.5

def calculate_metrics(
        model: Module,
        data_loader: DataLoader,
        device: torch.device,
        criterion: Callable,
        num_of_steps: int = -1,
        test_classes: Optional[List[int]] = None) -> Tuple[Union[float, int], float, float]:
    model.eval()

    correct_predictions, total_predictions = 0, 0
    total_loss = 0.0
    steps_completed = 0
    grads_average = 0.
    with torch.no_grad():
        for images, labels in data_loader:
            if num_of_steps > 0 and steps_completed >= num_of_steps:
                break

            steps_completed += 1

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            grads_average +=get_grad_norm(model.parameters())

            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs, 1)
            label_match = (predicted_labels == labels)

            if test_classes is not None:
                label_match = (predicted_labels == (labels - 1))
                test_classes_tensor = torch.tensor(test_classes).to(device)
                mask = (labels[:, None] == test_classes_tensor).any(dim=1)
                label_match = label_match[mask]

            total_predictions += label_match.size(0)
            correct_predictions += label_match.sum().item()

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions else 0
    loss = total_loss / steps_completed if steps_completed else 0.0
    grads_average = grads_average / steps_completed
    return accuracy, loss, grads_average
