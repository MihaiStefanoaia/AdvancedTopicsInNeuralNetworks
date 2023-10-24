from typing import Tuple, List
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST
from tqdm import tqdm


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def to_one_hot(x: Tensor) -> Tensor:
    return torch.eye(x.max() + 1)[x]


def collate(x) -> Tensor:
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise "Not supported yet"


def load_mnist(path: str = "./data", train: bool = True, pin_memory: bool = True):
    mnist_raw = MNIST(path, download=True, train=train)
    mnist_data = []
    mnist_labels = []
    for image, label in mnist_raw:
        tensor = torch.from_numpy(np.array(image))
        mnist_data.append(tensor)
        mnist_labels.append(label)

    mnist_data = collate(mnist_data).float()  # shape 60000, 28, 28
    mnist_data = mnist_data.flatten(start_dim=1)  # shape 60000, 784
    mnist_data /= mnist_data.max()  # min max normalize
    mnist_labels = collate(mnist_labels)  # shape 60000
    if train:
        mnist_labels = to_one_hot(mnist_labels)  # shape 60000, 10
    if pin_memory:
        return mnist_data.pin_memory(), mnist_labels.pin_memory()
    return mnist_data, mnist_labels


def forward_pass(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x @ w + b


def activation(x: Tensor):
    return x.sigmoid()


def activation_derivative(sig: Tensor):  # apply only to
    return sig * (1 - sig)


def deep_forward(data: Tensor, weights: List[Tensor], bias: List[Tensor]) -> List[Tensor]:
    activations = []
    for w_, b_ in zip(weights, bias):
        data = forward_pass(data, w_, b_)
        data = activation(data)
        activations.append(data)
    return activations


def backpropagate(activations: Tensor, error: Tensor, weights: Tensor, biases: Tensor):
    return (activations * (1 - activations)) * (error @ weights.T)


def backward_pass(data: Tensor, error: Tensor) -> Tuple[Tensor, Tensor]:
    delta_w = data.T @ error
    delta_b = error.mean(dim=0)  # On column
    return delta_w, delta_b


def train_batch(data: Tensor, expectation: Tensor, weights: List[Tensor], biases: List[Tensor], learning_rate: float):
    activations = deep_forward(data, weights, biases)
    errors = [activations[-1] - expectation]

    for i in range(1, len(activations))[::-1]:
        errors.append(backpropagate(activations[i-1], errors[-1], weights[i], biases[i]))
    errors.reverse()

    datas = [data] + activations[:-1]
    for i in range(len(weights)):
        d_w, d_b = backward_pass(datas[i], errors[i])
        weights[i] -= learning_rate * d_w
        biases[i] -= learning_rate * d_b

    return weights, biases

def train_epoch(data: Tensor, labels: Tensor, weights: List[Tensor], biases: List[Tensor], learning_rate: float, batch_size: int) \
                -> tuple[list[Tensor], list[Tensor]]:
    non_blocking = weights[0].device.type == 'cuda'
    for i in range(0, data.shape[0], batch_size):
        batch_data = data[i: i + batch_size].to(weights[0].device, non_blocking=non_blocking)
        batch_labels = labels[i: i + batch_size].to(weights[0].device, non_blocking=non_blocking)
        weights, biases = train_batch(batch_data, batch_labels, weights, biases, learning_rate)
    return weights, biases

def check(data: Tensor, labels: Tensor, w: List[Tensor], b: List[Tensor], batch_size: int) -> float:
    # Labels are not one hot encoded, because we do not need them as one hot.
    total_correct_predictions = 0
    total_len = data.shape[0]
    non_blocking = w[0].device.type == 'cuda'
    for i in range(0, total_len, batch_size):
        x = data[i: i + batch_size].to(w[0].device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w[0].device, non_blocking=non_blocking)
        predicted_distribution = deep_forward(x, w, b)[-1]
        # check torch.max documentation
        predicted_max_value, predicted_max_value_indices = torch.max(predicted_distribution, dim=1)
        # we check if the indices of the max value per line correspond to the correct label. We get a boolean mask
        # with True where the indices are the same, false otherwise
        equality_mask = predicted_max_value_indices == y
        # We sum the boolean mask, and get the number of True values in the mask. We use .item() to get the value out of
        # the tensor
        correct_predictions = equality_mask.sum().item()
        total_correct_predictions += correct_predictions

    return 100 * total_correct_predictions / total_len


def train(epochs: int = 1000,
          learning_rate: float = 0.0005,
          device: torch.device = get_default_device(),
          shape: List[int] = [784, 10],
          batch_size: int = 100,
          initialization_interval: Tuple[int, int] = (-1, 1),
          lr_decay_frequency: int = 50,
          lr_decay_amount: float = 0.9):
    if shape[0] != 784 or shape[-1] != 10:
        raise RuntimeError('Error: The first layer has to be 784 and the last layer has to be 10')
    weights = []
    biases = []
    for i in range(len(shape)-1):
        weights.append((initialization_interval[1] - initialization_interval[0]) * torch.rand((shape[i], shape[i+1]), device=device) - initialization_interval[1])
        biases.append(torch.zeros((1, shape[i+1]), device=device))
    pin_memory = device.type == 'cuda'
    data, labels = load_mnist(pin_memory=pin_memory)
    test_data, test_labels = load_mnist(train=False, pin_memory=pin_memory)
    batch_size_test = 500
    epochs = tqdm(range(epochs))
    for epoch in epochs:
        weights, biases = train_epoch(data, labels, weights, biases, learning_rate, batch_size)
        train_accuracy = check(data, torch.max(labels,dim=1)[1], weights, biases, batch_size_test)
        test_accuracy = check(test_data, test_labels, weights, biases, batch_size_test)
        loss = torch.nn.functional.cross_entropy(
                    deep_forward(data.to(weights[0].device, non_blocking=pin_memory), weights, biases)[-1],
                    labels.to(weights[0].device, non_blocking=pin_memory))
        epochs.set_postfix_str(f"train accuracy = {train_accuracy}%, test accuracy = {test_accuracy}%, loss = {loss}")
        if epoch % lr_decay_frequency == 0:
            learning_rate *= lr_decay_amount

if __name__ == '__main__':
    torch.set_printoptions(precision=4)
    # sh = [784, 10]
    sh = [784, 100, 10]
    # sh = [784, 16, 16, 10]
    # batch_sizes = [500, 1000, 2000, 5000, 10000]
    # learning_rates = [0.00001, 0.0005, 0.001]
    # for bs in batch_sizes:
    #     for lr in learning_rates:
            # print(f'learning rate = {lr}, batch size = {bs}')
    train(epochs=500, shape=sh, batch_size=2000, learning_rate=0.0005)  # chose the best one
    train(epochs=500, shape=sh, batch_size=2000, learning_rate=0.0005, device=torch.device('cpu'))  # chose the best one
    
