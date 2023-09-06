import torch.nn as nn
import timm
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from flexibility_nn.models.resnet import ResNet9, ResNet6, ResNet18X, ResNet3, ResNet20X, ResNet22X
from flexibility_nn.models.swin import SwinTransformer
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fbpca  # import the fbpca package
from torch import lobpcg
from torch import svd_lowrank



def CombinedRDKronFiLM(D, d, params, names, seed=0):
    rdkron = RoundedDoubleKron(D, d, params, names, seed=seed)
    FiLM = FiLMLazyRandom(D, d, params, names, seed=seed)

    return (rdkron + FiLM) * (1 / np.sqrt(2))


def approximate_orthogonal_matrix(size, M):
    matrix = torch.randn(size, M, device='cuda')
    U, _, _ = svd_lowrank(matrix, q=M)
    return U

def johnson_lindenstrauss_projection_matrix(size, M):
    # Generate a random Gaussian projection matrix
    projection_matrix = torch.randn(size, M, device="cuda") / np.sqrt(M)
    return projection_matrix
def power_iteration(matrix, num_iter=10):
    Q = torch.randn(matrix.shape[1], 1, device="cuda")
    for _ in range(num_iter):
        Q = torch.matmul(matrix.t(), torch.matmul(matrix, Q))
        Q, _ = torch.linalg.qr(Q.unsqueeze(-1))  # Reshape Q to be 2-dimensional before calling qr
        Q = Q.squeeze(-1)  # Reshape Q back to its original shape
    return Q

def approximate_orthogonal_matrix2(size, M, num_iter=10):
    matrix = torch.randn(size, M, device="cuda")
    U = power_iteration(matrix, num_iter)
    return U
def modified_gram_schmidt_torch(matrix):
    Q = torch.zeros_like(matrix)
    for i in range(matrix.shape[1]):
        q = matrix[:, i]
        for j in range(i):
            q = q - torch.dot(Q[:, j], matrix[:, i]) * Q[:, j]
        q = q / torch.linalg.norm(q)
        Q[:, i] = q
    return Q
def get_projection_matrix(size, M):
    covariance_matrix = torch.randn(size, size)
    eigenvectors = get_projection_matrix_eigenvectors(covariance_matrix, M)
    return eigenvectors

def get_projection_matrix_eigenvectors(covariance_matrix, M):
    # Initialize random guess vectors
    guess_vectors = torch.randn(covariance_matrix.size(0), M, device=covariance_matrix.device)

    # Compute the largest M eigenvalues and eigenvectors using LOBPCG
    eigenvalues, eigenvectors = lobpcg(covariance_matrix, guess_vectors, largest=True)

    # No need to sort since LOBPCG returns the largest eigenvalues and eigenvectors
    return eigenvectors

class CustomProjectionModel(nn.Module):
    def __init__(self, base_model, M):
        super(CustomProjectionModel, self).__init__()
        self.base_model = base_model
        # Get the total number of parameters in the base model (including biases)
        total_params = sum(p.numel() for p in base_model.parameters())
        # Create a fixed orthogonal projection matrix P
        #H = torch.randn(total_params, M, device="cuda")  # Create a random matrix H on GPU
        #P = modified_gram_schmidt_torch(H)
        #P = johnson_lindenstrauss_projection_matrix(total_params, M) # Use randomized SVD to compute an approximate orthogonal matrix P
        P = approximate_orthogonal_matrix(total_params, M)
        self.P = P.float().requires_grad_(False)


        # Create a learnable vector u
        if True:
            init_params = []
            for p in self.base_model.parameters():
                init_params.append(p.view(-1))
            init_params = torch.cat(init_params, dim=0).unsqueeze(1).to("cuda")
            print(f'Shape of P: {self.P.shape}')
            print(f'Shape of init_params: {init_params.shape}')
            print(f'Shape of P.T: {self.P.T.shape}')
            # Initialize the learnable vector u based on the initial parameters of the base model
            self.u = nn.Parameter(torch.matmul(self.P.T, init_params))

    def forward(self, x):
        # Calculate the projected parameters for the base model
        projected_params = torch.mm(self.P, self.u).view(-1)

        # Set the base model's parameters to the projected parameters
        idx = 0
        for p in self.base_model.parameters():
            numel = p.numel()
            p.data.copy_(projected_params[idx:idx + numel].view(p.size()))
            idx += numel

        # Run the forward pass on the base model
        s =self.base_model(x)
        return s

class CustomConvNet(nn.Module):
    def __init__(self, conv_layers_config, num_classes=10, num_filters=32, hidden_size=100):
        super(CustomConvNet, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i, config in enumerate(conv_layers_config):
            in_channels, kernel_size, stride, padding = config
            out_channels = num_filters
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv_layers.append(conv_layer)
            self.conv_layers.append(nn.ReLU(inplace=True))
            if True or i % 2==1 or  i==len(conv_layers_config)-1:
                self.conv_layers.append(nn.MaxPool2d(2, 2))
            num_filters *= 2
            in_channels = out_channels  # Update in_channels for the next layer

        self.flat_features = self._get_flat_features()
        self.fc1 = nn.Linear(self.flat_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, self.flat_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_flat_features(self):
        x = torch.randn(1, 3, 32, 32)
        for layer in self.conv_layers:
            x = layer(x)
        return x.view(-1).size(0)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, nonlinearity='relu'):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size * input_size *3

        activation = self.get_activation(nonlinearity)  # Get the activation function
        for hidden_size in hidden_sizes:  # This should work now since hidden_sizes is an iterable
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation,
            ])
            prev_size = hidden_size
        #layers.append(nn.Linear(prev_size, num_classes))
        self.layers = nn.Sequential(*layers)
        self.fc= nn.Linear(prev_size, num_classes)

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        rep=self.layers(x)
        return self.fc(rep)

    @staticmethod
    def get_activation(nonlinearity):
        if nonlinearity.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif nonlinearity.lower() == 'tanh':
            return nn.Tanh()
        elif nonlinearity.lower() == 'leakyrelu':
            return nn.LeakyReLU()
        elif nonlinearity.lower() == 'linear':
            return nn.Identity()  # Use Identity for linear (no activation)
        else:  # Default to ReLU
            return nn.ReLU()


def count_parameters(model: nn.Module) -> int:
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_parameters


def generate_conv_layers_config(num_layers, num_filters, kernel_size=3, stride=1, padding=1):
    conv_layers_config = [(3, kernel_size, stride, padding)]  # 1st conv layer with 3 input channels
    for i in range(1, num_layers):
        in_channels = num_filters * (2 ** (i - 1))
        conv_layers_config.append((in_channels, kernel_size, stride, padding))
    return conv_layers_config



def get_model(architecture: str, num_classes: int, input_size: int, hidden_layers: List[int],
                            nonlinearity: str, args: dict) \
        -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Returns the model, criterion, optimizer, and learning rate scheduler based on the given architecture and hyperparameters.

    :param args:
    :param lr: Learning rate
    :param momentum: Momentum (only used for SGD optimizer)
    :param weight_decay: Weight decay
    :param epochs: Number of training epochs
    :param architecture: Model architecture
    :param num_classes: Number of output classes
    :param input_size: Number of input features (only used for MLP)
    :param hidden_layers: List of hidden layer sizes (only used for MLP)
    :return: Tuple of model, criterion, optimizer, and learning rate scheduler
    """
    model_constructors = {
        "resnet3": ResNet3,
        "resnet6": ResNet6,
        "resnet9": ResNet9,
        "resnet18X": ResNet18X,
        "resnet20X": ResNet20X,
        "resnet22X": ResNet22X,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
        "mlp": MLP,
        'convnet': CustomConvNet,
        'swin': SwinTransformer
    }
    for i in range(8):
        model_constructors[f"efficientnet_b{i}"] = lambda: timm.create_model(f"efficientnet_b{i}", pretrained=False, num_classes = num_classes)

    if architecture in model_constructors:
        if architecture == "mlp":
            net = model_constructors[architecture](input_size, hidden_layers, num_classes, nonlinearity=nonlinearity)
        elif architecture == 'convnet':
            conv_layers_config = generate_conv_layers_config(args.num_layers, args.num_filters)

            net = CustomConvNet(conv_layers_config,
                                num_filters=args.num_filters, hidden_size=args.hidden_size, num_classes=num_classes)
        elif 'efficientnet_b' in architecture:
            net = model_constructors[architecture]()
        elif architecture =='swin':
            #ll = []
            #for i in args.num_layers:
            #    ll.append()
            net  = SwinTransformer(num_classes, 32,
                        #num_blocks_list=[4, 4], dims=[128, 128, 256],
                        num_blocks_list=args.swin_num_blocks_list, dims=args.swin_dims,
                        head_dim=args.swin_head_dim, patch_size=2, window_size=4,
                        emb_p_drop=0., trans_p_drop=0., head_p_drop=0.1)
        elif architecture =='resnet18' or architecture =='resnet34':
            net = model_constructors[architecture]()
            num_ftrs = net.fc.in_features  # Getting last layer's output features
            net.fc = nn.Linear(num_ftrs, num_classes)
        else:
            net = model_constructors[architecture](num_classes=num_classes, image_size=input_size)
    else:
        raise ValueError(
            "Invalid architecture. Choose from 'resnet9', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'efficientnet_b0' to 'efficientnet_b7', or 'mlp'.")
    return net


def get_optimizer(net, optimizer_choice, lr, weight_decay, momentum, total_steps):
    if optimizer_choice.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice.lower() == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # Default to SGD
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=total_steps // 3, T_mult=2, eta_min=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
    criterion = nn.CrossEntropyLoss()
    return optimizer, lr_scheduler, criterion
