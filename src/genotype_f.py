"""SNN neuroevolution thesis (manager).

Author:     vl
Date:       2025-07-02
Py Ver:     3.12

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ]


    """
# Standard library
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import snntorch as snn
import torch    
import torchvision
from rich.console import Console
from rich.traceback import install
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Global constants
SEED = 15

# Global functions
install(show_locals=True)
console = Console(width=180)
RNG = np.random.default_rng(SEED)
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Get device
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

# Warning Control
# Type Checking
# Type Aliases

# ANN Config
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # Normalization helps with FashionMNIST
])

train_data = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

def create_snn(layer_sizes, beta, dropout_rate=0.1):
    """
    Create SNN with variable number of layers and improved architecture
    """

    class SNNModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.layers = nn.ModuleList()
            self.lif_layers = nn.ModuleList()
            self.dropout_layers = nn.ModuleList()

            # Layer params
            input_size = 28 * 28
            num_of_classes = 10

            # Evolvable hidden layers
            for i, hidden_size in enumerate(layer_sizes):
                self.layers.append(nn.Linear(input_size, hidden_size))
                self.lif_layers.append(snn.Leaky(beta=beta))

                # Add dropout for regularization (except last layer)
                if i < len(layer_sizes) - 1:
                    self.dropout_layers.append(nn.Dropout(dropout_rate))
                else:
                    self.dropout_layers.append(nn.Identity())

                input_size = hidden_size

            # Output layer
            self.layers.append(nn.Linear(input_size, num_of_classes))
            self.lif_layers.append(snn.Leaky(beta=beta))

            # Improved weight initialization for SNNs
            self.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                # Xavier initialization with slight bias for SNNs
                torch.nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        def forward(self, x, num_steps=15):  # Increased time steps
            # Initialize membrane potentials for all layers
            mem_states = [lif.init_leaky() for lif in self.lif_layers]
            spk_out = 0

            for step in range(num_steps):
                current_input = x.view(x.size(0), -1)

                # Forward through all layers
                for i, (layer, lif, dropout) in enumerate(
                    zip(
                        self.layers,
                        self.lif_layers,
                        self.dropout_layers + [nn.Identity()],
                    )
                ):
                    current = layer(current_input)
                    spike, mem_states[i] = lif(current, mem_states[i])
                    if (
                        i < len(self.layers) - 1
                    ):  # Apply dropout except on output
                        spike = dropout(spike)
                    current_input = spike

                spk_out += current_input

            return spk_out / num_steps

    return SNNModel()


def create_individual_with_depth():
    """Create individual with improved parameter ranges and depth diversity."""
    # Bias toward mid-range depths (3-6 layers) but allow full range
    depth_weights = [0.05, 0.05, 0.05, 0.20, 0.25, 0.20, 0.10, 0.05, 0.05]  # Favor 4-6 layers more strongly
    num_hidden_layers = np.random.choice(range(1, 10), p=depth_weights) 

    # Create layer sizes with tapering (wider to narrower)
    layer_sizes = []
    base_size = 512

    for i in range(num_hidden_layers):
        # Gradually reduce size for deeper layers
        reduction_factor = 0.85**i
        size = max(64, int(base_size * reduction_factor + RNG.uniform(-30, 30)))
        layer_sizes.append(size)

    # Depth-adaptive parameters
    beta = RNG.uniform(0.6, 0.90)  # Wider beta range
    
    # Adjust learning rate based on depth - deeper networks need lower LR
    base_lr = RNG.uniform(0.0008, 0.005)
    lr = base_lr / (1 + num_hidden_layers * 0.03)  # Lower LR for deeper networks
    
    dropout = RNG.uniform(0.05, 0.3)  # Slightly higher dropout range

    # [num_hidden_layers, *layer_sizes, beta, lr, dropout]
    return [num_hidden_layers, *layer_sizes, beta, lr, dropout]


def bounded_mutation_with_depth(individual, mu=0, sigma=0.15, indpb=0.2):
    """Improved mutation with better depth exploration."""
    num_layers = int(individual[0])

    # Mutate number of layers (increased probability)
    if RNG.random() < 0.7:  # Increased from 0.15
        # Allow both adding and removing layers
        change = 1 if RNG.random() < 0.75 else -1 # Can remove, stay same, or add
        new_num_layers = max(2, min(8, num_layers + change))

        if new_num_layers > num_layers:
            # Add a new layer (smaller than previous)
            prev_size = individual[num_layers] if num_layers > 0 else 128
            new_layer_size = max(32, int(prev_size * 0.7 + RNG.uniform(-10, 10)))
            individual.insert(num_layers + 1, new_layer_size)
        elif new_num_layers < num_layers:
            # Remove a layer (remove from middle preferentially)
            if num_layers > 1:
                remove_idx = RNG.choice(range(1, min(num_layers + 1, len(individual) - 3)))
                individual.pop(remove_idx)

        individual[0] = new_num_layers
        num_layers = new_num_layers

    # Mutate layer sizes with smaller perturbations
    for i in range(1, num_layers + 1):
        if i < len(individual) - 3 and RNG.random() < indpb:
            individual[i] += RNG.normal(mu, sigma * 30)  # Smaller mutations
            individual[i] = max(32, min(512, individual[i]))  # Wider range

    # Mutate beta (wider range)
    if len(individual) > num_layers + 1 and RNG.random() < indpb:
        individual[num_layers + 1] += RNG.normal(mu, sigma * 0.08)
        individual[num_layers + 1] = max(0.5, min(0.9, individual[num_layers + 1]))

    # Mutate learning rate
    if len(individual) > num_layers + 2 and RNG.random() < indpb:
        individual[num_layers + 2] += RNG.normal(mu, sigma * 0.002)
        individual[num_layers + 2] = max(0.0005, min(0.01, individual[num_layers + 2]))

    # Mutate dropout
    if len(individual) > num_layers + 3 and RNG.random() < indpb:
        individual[num_layers + 3] += RNG.normal(mu, sigma * 0.05)
        individual[num_layers + 3] = max(0.0, min(0.3, individual[num_layers + 3]))

    return (individual,)


def crossover_with_depth(ind1, ind2):
    """Improved crossover for variable-length individuals with diversity preservation."""
    num_layers1 = int(ind1[0])
    num_layers2 = int(ind2[0])

    # Sometimes choose depth randomly to maintain diversity
    if RNG.random() < 0.3:  # 30% chance for random depth choice
        chosen_depth = RNG.choice([num_layers1, num_layers2])
    else:
        # Choose depth more intelligently (favor successful depths)
        if hasattr(ind1, "fitness") and hasattr(ind2, "fitness"):
            if ind1.fitness.valid and ind2.fitness.valid:
                if ind1.fitness.values[0] > ind2.fitness.values[0]:
                    chosen_depth = num_layers1
                else:
                    chosen_depth = num_layers2
            else:
                chosen_depth = RNG.choice([num_layers1, num_layers2])
        else:
            chosen_depth = RNG.choice([num_layers1, num_layers2])

    # Create new individuals
    new_ind1 = [chosen_depth]
    new_ind2 = [chosen_depth]

    # Blend layer sizes
    for i in range(chosen_depth):
        if i < num_layers1 and i < num_layers2:
            # Blend existing layers
            alpha = RNG.uniform(0.3, 0.7)  # Less extreme blending
            size1 = alpha * ind1[1 + i] + (1 - alpha) * ind2[1 + i]
            size2 = (1 - alpha) * ind1[1 + i] + alpha * ind2[1 + i]
        elif i < num_layers1:
            size1 = ind1[1 + i] + RNG.uniform(-10, 10)
            size2 = RNG.uniform(32, 256)
        elif i < num_layers2:
            size1 = RNG.uniform(32, 256)
            size2 = ind2[1 + i] + RNG.uniform(-10, 10)
        else:
            size1 = RNG.uniform(64, 256)
            size2 = RNG.uniform(64, 256)

        new_ind1.append(max(32, min(512, size1)))
        new_ind2.append(max(32, min(512, size2)))

    # Blend other parameters
    alpha = RNG.uniform(0.3, 0.7)
    beta1 = alpha * ind1[num_layers1 + 1] + (1 - alpha) * ind2[num_layers2 + 1]
    beta2 = (1 - alpha) * ind1[num_layers1 + 1] + alpha * ind2[num_layers2 + 1]
    lr1 = alpha * ind1[num_layers1 + 2] + (1 - alpha) * ind2[num_layers2 + 2]
    lr2 = (1 - alpha) * ind1[num_layers1 + 2] + alpha * ind2[num_layers2 + 2]
    dropout1 = (
        alpha * ind1[num_layers1 + 3] + (1 - alpha) * ind2[num_layers2 + 3]
    )
    dropout2 = (1 - alpha) * ind1[num_layers1 + 3] + alpha * ind2[
        num_layers2 + 3
    ]

    new_ind1.extend([beta1, lr1, dropout1])
    new_ind2.extend([beta2, lr2, dropout2])

    ind1[:] = new_ind1
    ind2[:] = new_ind2

    return ind1, ind2


def evaluate_model_with_depth(individual):
    """Improved evaluation with depth-adaptive training and better debugging."""
    try:
        # Parse individual
        num_layers = int(individual[0])
        layer_sizes = [
            max(64, min(768, round(individual[i])))
            for i in range(1, num_layers + 1)
        ]
        beta = max(0.5, min(0.95, float(individual[num_layers + 1])))
        lr = max(0.0003, min(0.008, float(individual[num_layers + 2])))
        dropout = max(0.0, min(0.4, float(individual[num_layers + 3])))

        # Debug: Print individual parameters
        console.print(f"Training: Layers={layer_sizes}, Beta={beta:.3f}, LR={lr:.4f}, Dropout={dropout:.3f}")

        model = create_snn(layer_sizes, beta, dropout)
        model.to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=5e-4,
            betas=(0.9, 0.999)
        )
        loss_fn = nn.CrossEntropyLoss()

        # Depth-adaptive training - deeper networks get more training
        max_batches = min(1200, 400 + num_layers * 100)
        
        # Extended training loop
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward()

            # Depth-adaptive gradient clipping
            if num_layers > 4:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            elif num_layers > 2:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            if batch_idx >= max_batches:
                break

        # More comprehensive evaluation
        model.eval()
        correct, total = 0, 0
        eval_loss = 0

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                outputs = model(data)
                loss = loss_fn(outputs, targets)
                eval_loss += loss.item()

                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += (pred == targets).sum().item()

                # Evaluate on more samples for better accuracy
                if total >= 3000:  # Increased from 2000
                    break

        accuracy = correct / total
        avg_train_loss = total_loss / num_batches
        avg_eval_loss = eval_loss / (batch_idx + 1)
        #commit
        # Debug: Print training details
        console.print(f"Training Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")
        console.print(f"Correct: {correct}/{total} = {accuracy:.4f}")

        # Small complexity penalty to encourage efficient architectures
        complexity_penalty = (num_layers * 0.001 + sum(layer_sizes) * 0.000005)
        final_fitness = accuracy

        console.print(f"Final Fitness: {final_fitness:.6f} (Accuracy: {accuracy:.4f}, Penalty: {complexity_penalty:.6f})")
        return (final_fitness,)

    except Exception as e:
        console.print(f"Evaluation ERROR: {str(e)}")

        return (-1.0,)