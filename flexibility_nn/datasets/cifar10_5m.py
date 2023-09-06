import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
class NpzDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        super(NpzDataset, self).__init__()

        # Combine all npz files into a single dataset
        self.data = []
        self.targets = []
        for file_path in file_paths:
            with np.load(file_path) as f:
                self.data.append(f['X'])
                self.targets.append(f['Y'])
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        self.transform = transform


    def __getitem__(self, index):
        # Convert data and target to torch.Tensor
        data = self.data[index]
        target = torch.from_numpy(np.array(self.targets[index])).long()

        if self.transform:
            data = self.transform(data)

       #data = torch.from_numpy(data).float()
        return data, target

    def __len__(self):
        return len(self.data)


class NpzDataset_n(Dataset):
    def __init__(self, directory, transform=None, num_files_to_load=50):
        super(NpzDataset_n, self).__init__()

        self.directory = directory
        self.transform = transform
        self.num_files_to_load = num_files_to_load

        # Store a list of all npz file paths
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]


        # Initialize indices for loaded data and loaded files
        self.data_idx = 0
        self.file_idx = 0
        self.total_samples = 0
        to_s = len(self.file_paths)
        i=0
        params_path = os.path.join(directory, 'params.json')
        if os.path.isfile(params_path):
            with open(params_path, 'r') as f:
                self.total_samples = json.load(f)['total_samples']

        else:
            self.total_samples = 0
            for file_path in self.file_paths:
                with np.load(file_path) as f:
                    self.total_samples += len(f['samples'])
            with open(params_path, 'w') as f:
                json.dump({'total_samples': self.total_samples}, f)

        np.random.shuffle(self.file_paths)

        # Initial data and target loading
        self.data = []
        self.targets = []
        self.load_new_data()

    def load_new_data(self):
        # Reset data lists
        self.data = []
        self.targets = []

        # Load new files
        for _ in range(self.num_files_to_load):
            if self.file_idx < len(self.file_paths):
                file_path = self.file_paths[self.file_idx]
                with np.load(file_path) as f:
                    self.data.append(f['samples'])
                    # Check if targets are one-hot encoded
                    targets = f['label']
                    if targets.ndim == 2 and targets.shape[1] > 1:
                        # If so, convert to class index using argmax
                        targets = np.argmax(targets, axis=-1)
                    self.targets.append(targets)
                self.file_idx += 1
            else:
                break

        # Concatenate all loaded data
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        # Reset data index
        self.data_idx = 0

    def __getitem__(self, index):
        # Check if all currently loaded data has been used
        if self.data_idx >= len(self.data):
            # Load new data
            self.load_new_data()

        # Get data and target
        data = self.data[self.data_idx]
        target = torch.from_numpy(np.array(self.targets[self.data_idx])).long()

        if self.transform:
            data = self.transform(data)

        # Increment data index
        self.data_idx += 1

        return data, target

    def __len__(self):
        return self.total_samples  # assuming all files have the same number of samples

class NpzDataset_n3(Dataset):
    def __init__(self, root, transform=None, target_transform = None, num_files_to_load=30,
                 random_labels: bool = False):
        super(NpzDataset_n3, self).__init__()

        self.directory = root
        self.target_transform = target_transform
        self.transform = transform
        self.num_files_to_load = num_files_to_load
        self.random_labels = random_labels
        # Store a list of all npz file paths
        self.file_paths = []
        for f_class in os.listdir(root):
            f_class_dir = os.path.join(root, f_class)
            if os.path.isdir(f_class_dir):
                for f in os.listdir(f_class_dir):
                    if f.endswith('.npz'):
                        self.file_paths.append(os.path.join(f_class_dir, f))

        # Initialize indices for loaded data and loaded files
        self.data_idx = 0
        self.file_idx = 0
        self.total_samples = 0
        to_s = len(self.file_paths)
        i=0
        params_path = os.path.join(root, 'params.json')
        if os.path.isfile(params_path):
            with open(params_path, 'r') as f:
                file_s =  json.load(f)
                self.total_samples =file_s['total_samples']
                self.num_classes = file_s['num_classes']
        self.total_samples = int(self.total_samples)
        np.random.seed(5)
        np.random.shuffle(self.file_paths)
        # Initial data and target loading
        self.data = []
        self.targets = []
        self.load_new_data()

    def load_new_data(self):
        # Reset data lists
        self.data = []
        self.targets = []

        # Load new files
        for _ in range(self.num_files_to_load):
            if self.file_idx < len(self.file_paths):
                file_path = self.file_paths[self.file_idx]
                with np.load(file_path) as f:
                    targets = f['label']
                    valid_indices = targets != 1000
                    targets = targets[valid_indices]
                    self.data.append(f['samples'][valid_indices])
                    # Check if targets are one-hot encoded
                    if targets.ndim == 2 and targets.shape[1] > 1:
                        # If so, convert to class index using argmax
                        targets = np.argmax(targets, axis=-1)
                    if self.random_labels:
                        targets = f['random_number'][valid_indices]
                        # del self.targets[:]

                    self.targets.append(targets)

                self.file_idx += 1
            else:
                break

        self.data = np.concatenate(self.data, axis=0)
        self.data = ((self.data * 127.5 + 128).clip(0, 255)).astype(np.uint8)
        self.targets = np.concatenate(self.targets, axis=0)



        indices = np.arange(self.data.shape[0])
        np.random.seed(5)

        #np.random.shuffle(indices)
        self.targets = self.targets[indices]
        self.data = self.data[indices]

        self.targets = torch.from_numpy(self.targets).long()
        self.data_idx = 0

    def __getitem__(self, index):
        # Check if all currently loaded data has been used
        #print (00000, index)
        if self.data_idx >= len(self.data):
            # Load new data
            self.load_new_data()
        # Get data and target
        data = self.data[self.data_idx]
        target = self.targets[self.data_idx]
        if self.transform:
            data = self.transform(data)
        #if self.target_transform:
        #    target = self.target_transform(target)

        self.data_idx += 1
        return data, target-1

    def __len__(self):
        return self.total_samples



import json

class NpzDataset_n2(Dataset):
    def __init__(self, root, transform=None, target_transform=None, chunk_size=2, random_labels: bool = False):
        super(NpzDataset_n2, self).__init__()

        self.directory = root
        self.transform = transform
        self.target_transform = target_transform
        self.chunk_size = chunk_size
        self.random_labels = random_labels

        params_path = os.path.join(root, 'params.json')
        if os.path.isfile(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                self.total_samples = int(params['total_samples'])
                self.index_map = params['index_map']
        else:
            raise ValueError(f"Params file not found at {params_path}")

        # Currently loaded chunk
        self.current_chunk = None
        self.current_chunk_start = 0

    def load_chunk(self, start_index):
        data_list = []
        targets_list = []
        end_index = min(start_index + self.chunk_size, len(self.index_map))
        for file_path, _ in self.index_map[start_index:end_index]:
            # Load the data from the file
            with np.load(file_path) as f:
                data = f['samples']
                targets = f['label']
                valid_indices = targets != 1000
                data = data[valid_indices]
                targets = targets[valid_indices]
                # Check if targets are one-hot encoded
                if targets.ndim == 2 and targets.shape[1] > 1:
                    # If so, convert to class index using argmax
                    targets = np.argmax(targets, axis=-1)
                if self.random_labels:
                    targets = f['random_number'][valid_indices]
                data_list.append(data)
                targets_list.append(targets)
        self.current_chunk = (np.concatenate(data_list, axis=0), np.concatenate(targets_list, axis=0))
        self.current_chunk_start = start_index

    def __getitem__(self, index):
        chunk_index = index // self.chunk_size
        index_within_chunk = index % self.chunk_size

        if chunk_index != self.current_chunk_start:
            self.load_chunk(chunk_index)

        data, target = self.current_chunk
        data = data[index_within_chunk]
        target = target[index_within_chunk]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)

        return data, target-1

    def __len__(self):
        return self.total_samples

class NpzDataset4(Dataset):
    def __init__(self, root, total_samples = 1000, transform=None, target_transform=None, random_labels: bool = False):
        super(NpzDataset4, self).__init__()

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.random_labels = random_labels

        #self.file_paths, self.num_classes, self.total_samples = self._get_file_info(total_samples)
        self.data = []
        self.targets = []
        self.update_file_info(total_samples)
        #self.load_new_data()



    def update_file_info(self, total_samples):
        self.file_paths, self.num_classes, self.total_samples = self._get_file_info(total_samples)
        self.current_file_idx = 0
        self.total_samples_pass = 0
        self.load_new_data()

    def _get_file_info(self, total_samples):
        params_path = os.path.join(self.root, 'params.json')
        if os.path.isfile(params_path):
            with open(params_path, 'r') as f:
                file_info = json.load(f)
                num_classes = int(file_info['num_classes'])

                file_paths = [(k, v) for k, v in file_info['files_paths'].items()]
                np.random.shuffle(file_paths)

                paths = []
                cumulative_samples = 0
                for path, num_samples in file_paths:
                    if cumulative_samples + int(num_samples) > total_samples:
                        break
                    paths.append(path)
                    cumulative_samples += int(num_samples)
        else:
            paths = []
            num_classes = 0
        #print (num_classes, cumulative_samples)
        return paths, num_classes, cumulative_samples

    def load_new_data(self):
        self.data = []
        self.targets = []
        for _ in range(self.total_samples):
            if self.current_file_idx >= len(self.file_paths):
                self.current_file_idx  =0
                break
            file_path = self.file_paths[self.current_file_idx]
            samples, targets = self._load_data_from_file(file_path)
            self.data.append(samples)
            self.targets.append(targets)
            self.current_file_idx += 1

        self._concatenate_and_transform_data()
        self.total_samples_pass +=len(self.data)

    def _load_data_from_file(self, file_path):
        with np.load(file_path) as f:
            targets = f['label']
            valid_indices = targets != 1000
            targets = targets[valid_indices]
            samples = f['samples'][valid_indices]
            if targets.ndim == 2 and targets.shape[1] > 1:
                targets = np.argmax(targets, axis=-1)
            if self.random_labels:
                targets = f['random_number'][valid_indices]
        return samples, targets

    def _concatenate_and_transform_data(self):
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        indices = np.arange(self.data.shape[0])
        self.targets = self.targets[indices]
        self.data = self.data[indices]
        self.data = ((self.data * 127.5 + 128).clip(0, 255)).astype(np.uint8)
        self.targets = torch.from_numpy(self.targets).long()
        self.current_data_idx = 0

    def __getitem__(self, _):
        # If all currently loaded data has been used, load new data
        print (self.current_data_idx , len(self.data))
        if self.current_data_idx >= len(self.data):
            self.load_new_data()

        data = self.data[self.current_data_idx]
        target = self.targets[self.current_data_idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)

        self.current_data_idx += 1
        return data, target-1

    def __len__(self):
        return self.total_samples
