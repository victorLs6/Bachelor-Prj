import torch
import torch.nn as nn
import snntorch as snn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import math

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

transform = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

def create_snn(layer_sizes, beta):
    """
    Create SNN with variable number of layers
    layer_sizes: list of hidden layer sizes (e.g., [128, 64, 32] for 3 hidden layers)
    """
    class SNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Build layers dynamically
            self.layers = nn.ModuleList()
            self.lif_layers = nn.ModuleList()
            
            # Input layer
            input_size = 28 * 28
            for i, hidden_size in enumerate(layer_sizes):
                self.layers.append(nn.Linear(input_size, hidden_size))
                self.lif_layers.append(snn.Leaky(beta=beta))
                input_size = hidden_size
            
            # Output layer
            self.layers.append(nn.Linear(input_size, 10))
            self.lif_layers.append(snn.Leaky(beta=beta))

        def forward(self, x, num_steps=10):
            # Initialize membrane potentials for all layers
            mem_states = [lif.init_leaky() for lif in self.lif_layers]
            spk_out = 0
            
            for _ in range(num_steps):
                current_input = x.view(x.size(0), -1)
                
                # Forward through all layers
                for i, (layer, lif) in enumerate(zip(self.layers, self.lif_layers)):
                    current = layer(current_input)
                    spike, mem_states[i] = lif(current, mem_states[i])
                    current_input = spike
                
                spk_out += current_input  # current_input is output spikes after last layer
            
            return spk_out / num_steps
    
    return SNNModel()

def create_individual_with_depth():
    """Create individual with variable depth (2-5 hidden layers)"""
    num_hidden_layers = random.randint(2, 5)  # 2 to 5 hidden layers
    
    # Create layer sizes
    layer_sizes = []
    for _ in range(num_hidden_layers):
        layer_sizes.append(random.uniform(32, 256))
    
    # Add other parameters
    beta = random.uniform(0.5, 0.99)
    lr = random.uniform(0.0001, 0.01)
    
    # Individual format: [num_layers, layer1_size, layer2_size, ..., beta, lr]
    individual = [num_hidden_layers] + layer_sizes + [beta, lr]
    return individual

def bounded_mutation_with_depth(individual, mu=0, sigma=0.2, indpb=0.2):
    """Custom mutation that handles variable-length individuals"""
    num_layers = int(individual[0])
    
    # Mutate number of layers (with lower probability)
    if random.random() < 0.05:  # 5% chance to change depth
        new_num_layers = max(2, min(5, num_layers + random.choice([-1, 1])))
        
        if new_num_layers > num_layers:
            # Add a new layer
            new_layer_size = random.uniform(32, 256)
            individual.insert(num_layers + 1, new_layer_size)
        elif new_num_layers < num_layers:
            # Remove a layer
            individual.pop(num_layers)
        
        individual[0] = new_num_layers
        num_layers = new_num_layers
    
    # Mutate layer sizes
    for i in range(1, num_layers + 1):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma * (256 - 32))
            individual[i] = max(32, min(256, individual[i]))
    
    # Mutate beta
    if random.random() < indpb:
        individual[num_layers + 1] += random.gauss(mu, sigma * (0.99 - 0.5))
        individual[num_layers + 1] = max(0.5, min(0.99, individual[num_layers + 1]))
    
    # Mutate learning rate
    if random.random() < indpb:
        individual[num_layers + 2] += random.gauss(mu, sigma * (0.01 - 0.0001))
        individual[num_layers + 2] = max(0.0001, min(0.01, individual[num_layers + 2]))
    
    return individual,

def crossover_with_depth(ind1, ind2):
    """Custom crossover for variable-length individuals"""
    # Simple approach: take depth from parent 1, layer sizes from blend of both
    num_layers1 = int(ind1[0])
    num_layers2 = int(ind2[0])
    
    # Choose depth from one parent
    chosen_depth = random.choice([num_layers1, num_layers2])
    
    # Create new individuals
    new_ind1 = [chosen_depth]
    new_ind2 = [chosen_depth]
    
    # Blend layer sizes (taking minimum of available layers)
    min_layers = min(num_layers1, num_layers2, chosen_depth)
    for i in range(chosen_depth):
        if i < min_layers:
            # Blend existing layers
            alpha = random.random()
            size1 = alpha * ind1[1 + i] + (1 - alpha) * ind2[1 + i]
            size2 = (1 - alpha) * ind1[1 + i] + alpha * ind2[1 + i]
        else:
            # Use random sizes for additional layers
            size1 = random.uniform(32, 256)
            size2 = random.uniform(32, 256)
        
        new_ind1.append(size1)
        new_ind2.append(size2)
    
    # Blend beta and lr
    alpha = random.random()
    beta1 = alpha * ind1[num_layers1 + 1] + (1 - alpha) * ind2[num_layers2 + 1]
    beta2 = (1 - alpha) * ind1[num_layers1 + 1] + alpha * ind2[num_layers2 + 1]
    lr1 = alpha * ind1[num_layers1 + 2] + (1 - alpha) * ind2[num_layers2 + 2]
    lr2 = (1 - alpha) * ind1[num_layers1 + 2] + alpha * ind2[num_layers2 + 2]
    
    new_ind1.extend([beta1, lr1])
    new_ind2.extend([beta2, lr2])
    
    ind1[:] = new_ind1
    ind2[:] = new_ind2
    
    return ind1, ind2

def evaluate_model_with_depth(individual):
    # Parse individual
    num_layers = int(individual[0])
    layer_sizes = [max(32, min(256, int(round(individual[i])))) 
                   for i in range(1, num_layers + 1)]
    beta = max(0.5, min(0.99, float(individual[num_layers + 1])))
    lr = max(0.0001, min(0.01, float(individual[num_layers + 2])))
    
    model = create_snn(layer_sizes, beta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop (same as before)
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        if batch_idx >= 20:
            break

    # Evaluation (same as before)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
            if total >= 1000:
                break
    
    accuracy = correct / total
    return (accuracy,)

def plot_individual(individual, ax, color='blue', alpha=0.7):
    """Plot a single individual's parameters as a bar chart"""
    num_layers = int(individual[0])
    layer_sizes = [individual[i] for i in range(1, num_layers + 1)]
    beta = individual[num_layers + 1]
    lr = individual[num_layers + 2]
    
    # Create parameter names and values for plotting
    params = [f'Layer{i+1}' for i in range(num_layers)] + ['Beta', 'LR×1000']
    values = layer_sizes + [beta * 100, lr * 1000]  # Scale for better visualization
    
    bars = ax.bar(params, values, color=color, alpha=alpha)
    ax.set_ylim(0, 300)
    ax.set_ylabel('Parameter Value')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

def plot_population(population, generation, min_fitness, max_fitness):
    """Plot population diversity and fitness distribution"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract first two layer sizes for plotting
    hidden1_vals = [ind[1] for ind in population]
    hidden2_vals = [ind[2] for ind in population]
    beta_vals = [ind[int(ind[0]) + 1] for ind in population]  # Beta position depends on num_layers
    lr_vals = [ind[int(ind[0]) + 2] for ind in population]    # LR position depends on num_layers
    fitness_vals = [ind.fitness.values[0] if ind.fitness.valid else 0 for ind in population]
    
    scatter1 = ax1.scatter(hidden1_vals, hidden2_vals, c=fitness_vals, 
                          cmap='viridis', s=100, alpha=0.7, vmin=min_fitness, vmax=max_fitness)
    ax1.set_xlabel('Hidden Layer 1 Size')
    ax1.set_ylabel('Hidden Layer 2 Size')
    ax1.set_title(f'Gen {generation}: Layer Sizes vs Fitness')
    ax1.grid(True, alpha=0.3)

    scatter2 = ax2.scatter(beta_vals, lr_vals, c=fitness_vals, 
                          cmap='viridis', s=100, alpha=0.7, vmin=min_fitness, vmax=max_fitness)
    ax2.set_xlabel('Beta (Leak Factor)')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title(f'Gen {generation}: Beta vs Learning Rate')
    ax2.grid(True, alpha=0.3)

    ax3.hist(fitness_vals, bins=max(3, len(population)//2), alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(np.mean(fitness_vals), color='red', linestyle='--', label=f'Mean: {np.mean(fitness_vals):.3f}')
    ax3.axvline(np.max(fitness_vals), color='green', linestyle='--', label=f'Max: {np.max(fitness_vals):.3f}')
    ax3.set_xlabel('Fitness (Accuracy)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Gen {generation}: Fitness Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    best_idx = np.argmax(fitness_vals)
    best_individual = population[best_idx]
    plot_individual(best_individual, ax4, color='gold')
    ax4.set_title(f'Gen {generation}: Best Individual (Acc: {fitness_vals[best_idx]:.3f})')
    
    cbar = plt.colorbar(scatter1, ax=ax4)
    cbar.set_label('Fitness (Accuracy)')
    
    plt.tight_layout()
    return fig

def plot_evolution_progress(logbook):
    """Plot evolution statistics over generations"""
    gen = logbook.select("gen")
    avg_fitness = logbook.select("avg")
    max_fitness = logbook.select("max")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(gen, avg_fitness, 'b-', label='Average Fitness', linewidth=2, marker='o')
    ax1.plot(gen, max_fitness, 'r-', label='Maximum Fitness', linewidth=2, marker='s')
    ax1.fill_between(gen, avg_fitness, max_fitness, alpha=0.2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness (Accuracy)')
    ax1.set_title('Evolution Progress: Fitness Over Generations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    improvement = np.diff(max_fitness)
    ax2.bar(gen[1:], improvement, alpha=0.7, color=['green' if x > 0 else 'red' for x in improvement])
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Improvement')
    ax2.set_title('Generation-to-Generation Improvement')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_architecture_evolution(populations_history):
    """Plot how architecture parameters evolved over time"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    param_names = ['Hidden Layer 1', 'Hidden Layer 2', 'Beta (Leak Factor)', 'Learning Rate']
    
    for idx, (ax, param_name) in enumerate(zip(axes.flat, param_names)):
        for gen, population_data in enumerate(populations_history):
            values = []
            fitness = []
            
            for ind_data in population_data:
                num_layers = int(ind_data[0])
                if idx == 0:  # Hidden Layer 1
                    values.append(ind_data[1])
                elif idx == 1:  # Hidden Layer 2
                    if num_layers >= 2:
                        values.append(ind_data[2])
                    else:
                        continue
                elif idx == 2:  # Beta
                    values.append(ind_data[num_layers + 1])
                elif idx == 3:  # Learning Rate
                    values.append(ind_data[num_layers + 2])
                
                fitness.append(ind_data[-1])  # Last element is fitness
            
            if values:  # Only plot if we have values
                scatter = ax.scatter([gen] * len(values), values, c=fitness, 
                                  cmap='viridis', alpha=0.6, s=50)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel(param_name)
        ax.set_title(f'Evolution of {param_name}')
        ax.grid(True, alpha=0.3)

    if len(populations_history) > 0:
        cbar = plt.colorbar(scatter, ax=axes.flat[-1])
        cbar.set_label('Fitness (Accuracy)')
    
    plt.tight_layout()
    return fig

# Create DEAP classes and toolbox
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# FIXED: Properly wrap the individual creation to return creator.Individual
def create_individual():
    """Create an individual and wrap it in creator.Individual"""
    ind_data = create_individual_with_depth()
    return creator.Individual(ind_data)

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_model_with_depth)
toolbox.register("mate", crossover_with_depth)
toolbox.register("mutate", bounded_mutation_with_depth, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def print_best_individual(individual):
    num_layers = int(individual[0])
    layer_sizes = [int(round(individual[i])) for i in range(1, num_layers + 1)]
    beta = individual[num_layers + 1]
    lr = individual[num_layers + 2]
    
    print(f"\nBest individual: {individual}")
    print(f"Final accuracy: {individual.fitness.values[0]:.4f}")
    
    # Build architecture string
    arch_str = "784"
    for size in layer_sizes:
        arch_str += f" → {size}"
    arch_str += " → 10"
    
    print(f"Architecture: {arch_str}")
    print(f"Depth: {num_layers} hidden layers")
    print(f"Beta: {beta:.3f}, Learning Rate: {lr:.6f}")

def run_evolution():
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    
    populations_history = []

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Store population history with fitness values
    pop_data = []
    for ind in pop:
        ind_data = list(ind) + [ind.fitness.values[0]]
        pop_data.append(ind_data)
    populations_history.append(pop_data)
    
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    print(logbook.stream)
    
    for gen in range(1, 21):  
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5: 
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.3: 
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        
        # Store population history with fitness values
        pop_data = []
        for ind in pop:
            ind_data = list(ind) + [ind.fitness.values[0]]
            pop_data.append(ind_data)
        populations_history.append(pop_data)
        
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

    print("\nGenerating visualizations...")

    fig1 = plot_evolution_progress(logbook)
    plt.savefig('evolution_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    min_fitness = min(logbook.select('min'))
    max_fitness = max(logbook.select('max'))
    
    plt.figure(figsize=(18, 12))
    for i in range(min(12, len(populations_history))):
        plt.subplot(3, 4, i+1)
        pop_data = []
        for ind_data in populations_history[i]:
            ind = creator.Individual(ind_data[:-1])  # Exclude fitness from individual data
            ind.fitness.values = (ind_data[-1],)     # Set fitness
            pop_data.append(ind)

        fitness_vals = [ind.fitness.values[0] for ind in pop_data]
        hidden1_vals = [ind[1] for ind in pop_data]
        hidden2_vals = [ind[2] for ind in pop_data]
        
        scatter = plt.scatter(hidden1_vals, hidden2_vals, c=fitness_vals, 
                            cmap='viridis', s=60, alpha=0.7, vmin=min_fitness, vmax=max_fitness)
        plt.xlabel('Hidden1')
        plt.ylabel('Hidden2')
        plt.title(f'Gen {i}')
        plt.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=plt.gca())
    plt.tight_layout()
    plt.savefig('population_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig3 = plot_architecture_evolution(populations_history)
    plt.savefig('architecture_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print_best_individual(hof[0])

if __name__ == '__main__':
    run_evolution()