import snntorch as snn
import torch

# Training Parameters
batch_size=128
data_path='/tmp/data/mnist'
num_classes = 10  # MNIST has 10 output classes

# Torch Variables
dtype = torch.float

from torchvision import datasets, transforms

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

# temporary dataloader if MNIST service is unavailable
!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
!tar -zxvf MNIST.tar.gz

mnist_train = datasets.MNIST(root = './', train=True, download=True, transform=transform)

# from snntorch import utils

# subset = 10
# mnist_train = utils.data_subset(mnist_train, subset)

# print(f"The size of mnist_train is {len(mnist_train)}")
# The size of mnist_train is 6000