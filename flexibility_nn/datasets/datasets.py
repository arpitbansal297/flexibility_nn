import os
from typing import Tuple, List
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import torch
import numpy as np
from torch.utils.data import random_split
import json
# Removed unused imports
from flexibility_nn.datasets.cifar10_5m import NpzDataset, NpzDataset_n, NpzDataset_n2, NpzDataset4
import math
import torch
from pathlib import Path
import functools
from torch.utils.data import DataLoader, Dataset
from flexibility_nn.datasets.utils import DataLoaderWithPrefetch
from flexibility_nn.utils import npz_to_jpeg, copy_images

# Datasets configurations
DATASET_CONFIGS = {
    "cifar100": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761], "size": 32},
    "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010], "size": 32},
    "cinic10": {"mean": [0.4789, 0.4723, 0.4305], "std": [0.2421, 0.2383, 0.2587], "size": 32},
    "inaturalist": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "size": 224},
    "imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "size": 32},
    "imagenet21k": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "size": 64},
    "tiny-imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "size": 224},
    # Added additional datasets here
    "cifar10_5m": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010], "size": 32},
    "cifar10_sampled": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010], "size": 32},
    "imagenet_sampled": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "size": 32},

}



def load_permuted_dataset(root, train=True, transform=None, target_transform=None):
    # Choose the appropriate path based on the train flag

    path_s = 'cifar10_rand_labels/permuted_train.pth' if train else 'cifar10_rand_labels/permuted_test.pth'
    path = root / Path(path_s)
    images, labels = torch.load(path)

    # If a transform is provided, apply it to the images
    if transform:
        transformed_images = []
        for img in images:
            pil_img = transforms.ToPILImage()(img)
            transformed_img = transform(pil_img)
            transformed_images.append(transformed_img)
        images = torch.stack(transformed_images)

    tensor_dataset = torch.utils.data.TensorDataset(images, labels)
    return tensor_dataset



def load_permuted_dataset_rand_labels(root, train=True, transform=None, target_transform=None):
    # Choose the appropriate path based on the train flag
    path_s = 'cifar10_rand_labels/permuted_train.pth' if train else 'cifar10_rand_labels/permuted_test.pth'
    path = root / Path(path_s)
    images, labels = torch.load(path)

    # If a transform is provided, apply it to the images
    if transform:
        transformed_images = []
        for img in images:
            pil_img = transforms.ToPILImage()(img)
            transformed_img = transform(pil_img)
            transformed_images.append(transformed_img)
        images = torch.stack(transformed_images)

    tensor_dataset = torch.utils.data.TensorDataset(images, labels)
    return tensor_dataset




def get_cifar10_options_random_input(data_dir, transform_train, transform_test):
    return {
        "dataset_class": load_permuted_dataset,
        "trainset": {"root": data_dir, "train": True, "transform": transform_train},
        "testset": {"root": data_dir, "train": False, "transform": transform_test},
        "old_num_classes": 10,
        "testset_org": {"root": data_dir, "transform": transform_test, "train": False},
        'make_target_transform': False
    }


def get_cifar10_options_random_label(data_dir, transform_train, transform_test):
    return {
        "dataset_class": load_permuted_dataset_rand_labels,
        "trainset": {"root": data_dir, "train": True, "transform": transform_train},
        "testset": {"root": data_dir, "train": False, "transform": transform_test},
        "old_num_classes": 10,
        "testset_org": {"root": data_dir, "transform": transform_test, "train": False},
        'make_target_transform': False
    }

def get_cifar10_options(data_dir, transform_train, transform_test):
    return {
        "dataset_class": datasets.CIFAR10,
        "trainset": {"root": data_dir, "train": True, "download": True, "transform": transform_train},
        "testset": {"root": data_dir, "train": False, "download": True, "transform": transform_test},
        "old_num_classes": 10,
        "testset_org": {"root": data_dir, "transform": transform_test, "train": False},
        'make_target_transform': False,
        "num_classes":10,
    }





def get_cifar100_options(data_dir, transform_train, transform_test):
    return {
        "dataset_class": datasets.CIFAR100,
        "trainset": {"root": data_dir, "train": True, "download": True, "transform": transform_train},
        "testset": {"root": data_dir, "train": False, "download": True, "transform": transform_test},
        "old_num_classes": 100,
        "testset_org": {"root": data_dir, "transform": transform_test, "train": False},
        'make_target_transform': True
    }


def get_imagenet_options(data_dir, transform_train, transform_test):
    return {
        "dataset_class": ImageFolder,
        "trainset": {"root": data_dir / "train", "transform": transform_train},
        "testset": {"root": "/imagenet/val", "transform": transform_test},
        "testset_org": {"root": "/imagenet/val", "transform": transform_test},
        "old_num_classes": 1_000,
        "num_classes":1_000,
        'make_target_transform': True,
    }


def get_imagenet21k_options(data_dir, transform_train, transform_test):
    return {
        "dataset_class": ImageFolder,
        "trainset": {"root": data_dir / "imagenet21k_train", "transform": transform_train},
        "testset": {"root": data_dir / "imagenet21k_val", "transform": transform_test},
        "testset_org": {"root": data_dir / "imagenet21k_val", "transform": transform_test},
        "old_num_classes": 21_000,
        'make_target_transform': True
    }





def get_transforms(dataset: str, augmenation: bool = False) -> Tuple[transforms.Compose, transforms.Compose, int]:
    """
    Get transforms for the specified dataset.

    Args:
        dataset: Dataset name.
        augmenation: If True, apply augmentation on train dataset.

    Returns:
        Tuple of training and testing transforms.
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError("Invalid dataset. Check the datasets listed in the DATASETS_CONFIG.")

    mean = DATASET_CONFIGS[dataset]["mean"]
    std = DATASET_CONFIGS[dataset]["std"]
    size = DATASET_CONFIGS[dataset]["size"]

    common_transforms_org = [ transforms.ToTensor(), transforms.Resize(size,  antialias=True), transforms.CenterCrop(size),transforms.Normalize(mean, std)]
    common_transforms = common_transforms_org
    if False and  dataset == 'imagenet_sampled':
        common_transforms = [transforms.Lambda(lambda img: np.transpose(img, (1, 2, 0)))] + common_transforms
    if augmenation:
        train_transforms = common_transforms + [transforms.RandomCrop(size, padding=4),
                                                transforms.RandomHorizontalFlip()]
    else:
        train_transforms = common_transforms

    test_transform =common_transforms_org
    transform_train = transforms.Compose(train_transforms)
    transform_test = transforms.Compose(test_transform)

    return transform_train, transform_test, size


def generate_random_labels(trainset: torch.utils.data.Dataset, testset: torch.utils.data.Dataset, num_classes: int) -> \
Tuple[List[int], List[int]]:
    """
    Replace original labels with random ones in both training and testing datasets

    Args:
        trainset: The training dataset.
        testset: The testing dataset.
        num_classes: Number of classes in the datasets.

    Returns:
        Updated labels for trainset and testset
    """
    train_random_labels = np.random.randint(0, num_classes, len(trainset))
    test_random_labels = np.random.randint(0, num_classes, len(testset))
    return list(train_random_labels), list(test_random_labels)


def randomize_input_data(trainset: torch.utils.data.Dataset, testset: torch.utils.data.Dataset) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Replace original input data with random ones in both training and testing datasets

    Args:
        trainset: The training dataset.
        testset: The testing dataset.

    Returns:
        Updated trainset and testset with randomized input data.
    """
    num_channels = trainset[0][0].shape[0]  # Number of channels in the image (3 for RGB)

    for idx in range(len(trainset)):
        trainset.data[idx] = torch.rand(num_channels, IMAGE_SIZE, IMAGE_SIZE)

    for idx in range(len(testset)):
        testset.data[idx] = torch.rand(num_channels, IMAGE_SIZE, IMAGE_SIZE)

    return trainset, testset




def bin_labels(labels, old_num_classes, new_num_classes):
    # This will map each old class to a new class. We use floor division so that
    # contiguous ranges of old classes map to the same new class.
    class_mapping = torch.arange(old_num_classes) // (old_num_classes // new_num_classes)

    # Now we just index into class_mapping with the original labels to get the new labels.
    # The view(-1) operation is necessary because PyTorch doesn't allow 1-D tensors to be
    # indexed with 1-D tensors, so we need to add an extra dimension.
    return class_mapping[labels]


def load_num_sample(root):
    params_path = os.path.join(root, 'params.json')
    if os.path.isfile(params_path):
        with open(params_path, 'r') as f:
            total_samples = json.load(f)['total_samples']
    return total_samples

class GaussianRandomDataset(Dataset):
    def __init__(self, dataset, gaussian = False, random_label = False, num_of_labels=1000):
        self.dataset = dataset
        self.input_shape = None
        self.gaussian = gaussian
        self.all_indexes = {}
        self.random_label = random_label
        self.num_of_labels = num_of_labels

    def __getitem__(self, index):
        data, target = self.dataset[index]

        # Generate a Gaussian random array of the same shape as the data
        random_state = np.random.RandomState(index)  # Use the index as the seed

        if self.random_label:
            target = random_state.randint(0, self.num_of_labels)
        else:
            # Convert the data to a PyTorch tensor and get its shape
            if self.input_shape is None:
                if not isinstance(data, torch.Tensor):
                    data_tensor = transforms.ToTensor()(data)
                else:
                    data_tensor = data
                self.input_shape = data_tensor.shape

            if self.gaussian:
                data = torch.from_numpy(random_state.normal(0., 1., self.input_shape)).float()
            else:
                data_reshpaed = data.reshape((-1,))
                per = torch.from_numpy(random_state.permutation(math.prod(self.input_shape)))
                data =  data_reshpaed[per].reshape(self.input_shape).float()


        return data, target

    def __len__(self):
        return len(self.dataset)


def load_data(transform_train: transforms.Compose, transform_test: transforms.Compose, batch_size: int,
              dataset: str = "cifar10", data_dir: str = "data", workers: int = 8, random_labels: bool = False,
              random_input: bool = False, num_new_classes: int = None) -> Tuple[
    DataLoader, DataLoader, DataLoader, int, List[int]]:
    data_dir = Path(data_dir)

    if dataset == "cifar10":

        #if  random_input:
        #    options = get_cifar10_options_random_input(data_dir, transform_train, transform_test)
        #elif random_labels:
        #    options = get_cifar10_options_random_label(data_dir, transform_train, transform_test)
        #if:
        options = get_cifar10_options(data_dir, transform_train, transform_test)

    elif dataset == "cifar100":
        options = get_cifar100_options(data_dir, transform_train, transform_test)
    elif dataset == "imagenet":
        options = get_imagenet_options(data_dir, transform_train, transform_test)
    elif dataset == "imagenet21k":
        options = get_imagenet21k_options(data_dir, transform_train, transform_test)
    else:
        raise ValueError("Invalid dataset. Available options are: cifar10, cifar100, imagenet, imagenet21k")

    if dataset =='imagenet_sampled':
        copy_images(options['new_root'], options['trainset']['root'], 10_000, 512, random_dest = random_labels)
        copy_images(options['new_root'], options['testset']['root'], 10_000, 512, random_dest = random_labels)
    if num_new_classes is not None:
        options['num_classes'] = num_new_classes

    if 'make_target_transform' in options:
        bin_labels_c = functools.partial(bin_labels,old_num_classes=options['old_num_classes'], new_num_classes=options['num_classes'])
        options['trainset']['target_transform'] = transforms.Lambda(bin_labels_c)
        options['testset']['target_transform'] = transforms.Lambda(bin_labels_c)
        options['testset_org']['target_transform'] = transforms.Lambda(bin_labels_c)
    dataset_class = options["dataset_class"]
    if 'random_labels' in options:
        options['trainset']['random_labels'] = random_labels
    test_classes = None
    if 'test_classes' in options:
        test_classes = options['test_classes']
    trainset = dataset_class(**options["trainset"])
    if 'split_test' in options:
        test_size = options['testset']['test_size']
        dataset_size = len(trainset)
        train_size = dataset_size - test_size
        trainset, testset = random_split(trainset, [train_size, test_size])
    else:
        testset = dataset_class(**options["testset"])
    if 'testset_org_class' in options:
        test_org_class= options['testset_org_class']
    else:
        test_org_class = dataset_class
    testset_org = test_org_class(**options["testset_org"])
    num_classes = options.get("num_classes")
    if random_input or random_labels:
        print ('Random Labels or Input')
        # Replace the datasets with Gaussian random datasets
        trainset = GaussianRandomDataset(trainset, random_label=random_labels)
        testset = GaussianRandomDataset(testset, random_label=random_labels)
    if False and random_labels:
        train_random_labels = torch.randint(0, num_classes, (len(trainset),))
        test_random_labels = torch.randint(0, num_classes, (len(testset),))
        trainset.targets = train_random_labels.tolist()
        testset.targets = test_random_labels.tolist()
        print ('RANDAOM LABELS _!!!!!!')

    trainloader = DataLoader(trainset, batch_size=batch_size,  persistent_workers = False, shuffle=False, num_workers=workers, pin_memory = False)
    testloader = DataLoader(testset, batch_size=512,  persistent_workers = True, shuffle=False, num_workers=workers,  pin_memory = True)
    testloader_org = DataLoader(testset_org, batch_size=512,  persistent_workers = True, shuffle=False, num_workers=workers,  pin_memory = True)
    return trainloader, testloader, testloader_org, num_classes, test_classes




class DataPrefetcher():
    def __init__(self, dataloader, img_shape, device):
        self.dataloader = dataloader
        self._len = len(dataloader)
        self.device = device
        torch.cuda.device(device)
        self.stream = torch.cuda.Stream()
        self.img_shape = img_shape

    def prefetch(self):
        try:
            self.next_video, self.next_label = next(self.dl_iter)
        except StopIteration:
            self.next_video = None
            self.next_label = None
            return
        with torch.cuda.stream(self.stream):
            self.next_label = self.next_label.to(self.device, non_blocking=True)
            self.next_video = self.next_video.to(self.device, non_blocking=True)

            self.next_video = self.next_video.float()
            #self.next_video = torch.nn.functional.interpolate(
            #    input=self.next_video,
            #    size=self.img_shape,
            #    mode="trilinear",
            #    align_corners=False,
            #    )

    def __iter__(self):
        self.dl_iter = iter(self.dataloader)
        self.prefetch()
        return self

    def __len__(self):
        return self._len

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        video = self.next_video
        label = self.next_label

        if video is None or label is None:
            raise StopIteration

        video.record_stream(torch.cuda.current_stream())
        label.record_stream(torch.cuda.current_stream())
        self.prefetch()
        return video, label


class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        first = True
        for batch in self.loader:
            with torch.cuda.stream(self.stream):  # stream - parallel
                self.next_input = batch[0].cuda(non_blocking=True) # note - (0-1) normalization in .ToTensor()
                self.next_target = batch[1].cuda(non_blocking=True)

            if not first:
                yield input, target  # prev
            else:
                first = False

            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target

            # Ensures that the tensor memory is not reused for another tensor until all current work queued on stream are complete.
            input.record_stream(torch.cuda.current_stream())
            target.record_stream(torch.cuda.current_stream())

        # final batch
        yield input, target

        # cleaning at the end of the epoch
        del self.next_input
        del self.next_target
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    def set_epoch(self, epoch):
        self.loader.sampler.set_epoch(epoch)



