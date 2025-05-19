import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import random


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    return correct / total


def mutate(model, mutation_rate=0.02):
    child = copy.deepcopy(model)
    with torch.no_grad():
        for param in child.parameters():
            noise = torch.randn_like(param) * mutation_rate
            param.add_(noise)
    return child

def select(population, fitnesses, num_best):
    sorted_indices = np.argsort(fitnesses)[::-1]
    return [population[i] for i in sorted_indices[:num_best]]


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='data/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


sample_data = next(iter(train_loader))
train_subset = [(sample_data[0], sample_data[1])]
subset_loader = DataLoader(train_subset, batch_size=1000)


population_size = 10
num_generations = 80
num_elites = 2

population = [MLP() for _ in range(population_size)]


for gen in range(num_generations):
    fitnesses = [evaluate(ind, subset_loader) for ind in population]
    print(f"Generation {gen+1} - Best Accuracy: {max(fitnesses):.4f}")

   
    elites = select(population, fitnesses, num_elites)

   
    new_population = elites.copy()
    while len(new_population) < population_size:
        parent = random.choice(elites)
        child = mutate(parent)
        new_population.append(child)

    population = new_population


best_model = select(population, [evaluate(ind, subset_loader) for ind in population], 1)[0]
test_dataset = datasets.MNIST(root='data/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000)
test_acc = evaluate(best_model, test_loader)
print(f"Final Test Accuracy: {test_acc:.4f}")