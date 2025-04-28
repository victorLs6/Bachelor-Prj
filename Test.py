import torch
import torch.nn as nn
import snntorch as snn
from torchvision import datasets, transforms
from deap import base, creator, tools
import random
import numpy as np

# ----------------------
# Spiking Neural Network
# ----------------------
class SimpleSNN(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.fc = nn.Linear(28*28, 10, bias=False)
        self.lif = snn.Leaky(beta=0.95)
        if weights is not None:
            self.set_weights(weights)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        mem = self.lif.init_leaky()
        spk_rec = []
        for step in range(10):  # Simulate 10 timesteps
            spk, mem = self.lif(self.fc(x), mem)
            spk_rec.append(spk)
        return torch.stack(spk_rec).sum(dim=0)

    def set_weights(self, flat_weights):
        with torch.no_grad():
            w = torch.tensor(flat_weights, dtype=torch.float32).view(self.fc.weight.shape)
            self.fc.weight.copy_(w)

# ----------------------
# Dataset
# ----------------------
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
sample_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)

# ----------------------
# DEAP Setup
# ----------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
NUM_PARAMS = 28 * 28 * 10

toolbox.register("attr_float", lambda: random.uniform(-0.5, 0.5))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NUM_PARAMS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ----------------------
# Fitness Function
# ----------------------
def evaluate(individual):
    model = SimpleSNN(weights=individual)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in sample_loader:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            break  # Only one batch for speed (tweak later)

    return correct / total,  # Return as tuple

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# ----------------------
# Evolution Loop
# ----------------------
pop = toolbox.population(n=10)
NGEN = 5

for gen in range(NGEN):
    print(f"Generation {gen}")
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
        toolbox.mutate(child1)
        toolbox.mutate(child2)
        del child1.fitness.values, child2.fitness.values

    for ind in offspring:
        ind.fitness.values = toolbox.evaluate(ind)

    pop[:] = offspring

top_ind = tools.selBest(pop, 1)[0]
print("Best fitness:", top_ind.fitness.values[0])
