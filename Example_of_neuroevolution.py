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
            
            # Input layer
            input_size = 28 * 28
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
            self.layers.append(nn.Linear(input_size, 10))
            self.lif_layers.append(snn.Leaky(beta=beta))

        def forward(self, x, num_steps=15):  # Increased time steps
            # Initialize membrane potentials for all layers
            mem_states = [lif.init_leaky() for lif in self.lif_layers]
            spk_out = 0
            
            for step in range(num_steps):
                current_input = x.view(x.size(0), -1)
                
                # Forward through all layers
                for i, (layer, lif, dropout) in enumerate(zip(self.layers, self.lif_layers, self.dropout_layers + [nn.Identity()])):
                    current = layer(current_input)
                    spike, mem_states[i] = lif(current, mem_states[i])
                    if i < len(self.layers) - 1:  # Apply dropout except on output
                        spike = dropout(spike)
                    current_input = spike
                
                spk_out += current_input
            
            return spk_out / num_steps
    
    return SNNModel()

def create_individual_with_depth():
    """Create individual with improved parameter ranges"""
    num_hidden_layers = random.randint(2, 4)  # Reduced max depth for stability
    
    # Create layer sizes with tapering (wider to narrower)
    layer_sizes = []
    base_size = random.uniform(128, 256)  # Start with wider layers
    
    for i in range(num_hidden_layers):
        # Gradually reduce size for deeper layers
        reduction_factor = 0.7 ** i
        size = max(32, base_size * reduction_factor + random.uniform(-20, 20))
        layer_sizes.append(size)
    
    beta = random.uniform(0.7, 0.95)  # Narrower beta range
    lr = random.uniform(0.001, 0.005)  # Narrower LR range
    dropout = random.uniform(0.0, 0.2)  # Add dropout parameter
    
    individual = [num_hidden_layers] + layer_sizes + [beta, lr, dropout]
    return individual

def bounded_mutation_with_depth(individual, mu=0, sigma=0.15, indpb=0.15):
    """Improved mutation with smaller perturbations"""
    num_layers = int(individual[0])
    
    # Mutate number of layers (lower probability)
    if random.random() < 0.05:  # Reduced probability
        new_num_layers = max(2, min(4, num_layers + random.choice([-1, 1])))
        
        if new_num_layers > num_layers:
            # Add a new layer (smaller than previous)
            prev_size = individual[num_layers] if num_layers > 0 else 128
            new_layer_size = max(32, prev_size * 0.7 + random.uniform(-10, 10))
            individual.insert(num_layers + 1, new_layer_size)
        elif new_num_layers < num_layers:
            individual.pop(num_layers)
        
        individual[0] = new_num_layers
        num_layers = new_num_layers
    
    # Mutate layer sizes with smaller perturbations
    for i in range(1, num_layers + 1):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma * 30)  # Smaller mutations
            individual[i] = max(32, min(256, individual[i]))
    
    # Mutate beta
    if random.random() < indpb:
        individual[num_layers + 1] += random.gauss(mu, sigma * 0.05)
        individual[num_layers + 1] = max(0.7, min(0.95, individual[num_layers + 1]))
    
    # Mutate learning rate
    if random.random() < indpb:
        individual[num_layers + 2] += random.gauss(mu, sigma * 0.001)
        individual[num_layers + 2] = max(0.001, min(0.005, individual[num_layers + 2]))
    
    # Mutate dropout
    if random.random() < indpb:
        individual[num_layers + 3] += random.gauss(mu, sigma * 0.05)
        individual[num_layers + 3] = max(0.0, min(0.2, individual[num_layers + 3]))
    
    return individual,

def crossover_with_depth(ind1, ind2):
    """Improved crossover for variable-length individuals"""
    num_layers1 = int(ind1[0])
    num_layers2 = int(ind2[0])
    
    # Choose depth more intelligently (favor successful depths)
    if hasattr(ind1, 'fitness') and hasattr(ind2, 'fitness'):
        if ind1.fitness.valid and ind2.fitness.valid:
            if ind1.fitness.values[0] > ind2.fitness.values[0]:
                chosen_depth = num_layers1
            else:
                chosen_depth = num_layers2
        else:
            chosen_depth = random.choice([num_layers1, num_layers2])
    else:
        chosen_depth = random.choice([num_layers1, num_layers2])
    
    # Create new individuals
    new_ind1 = [chosen_depth]
    new_ind2 = [chosen_depth]
    
    # Blend layer sizes
    for i in range(chosen_depth):
        if i < num_layers1 and i < num_layers2:
            # Blend existing layers
            alpha = random.uniform(0.3, 0.7)  # Less extreme blending
            size1 = alpha * ind1[1 + i] + (1 - alpha) * ind2[1 + i]
            size2 = (1 - alpha) * ind1[1 + i] + alpha * ind2[1 + i]
        elif i < num_layers1:
            size1 = ind1[1 + i] + random.uniform(-10, 10)
            size2 = random.uniform(32, 128)
        elif i < num_layers2:
            size1 = random.uniform(32, 128)
            size2 = ind2[1 + i] + random.uniform(-10, 10)
        else:
            size1 = random.uniform(64, 128)
            size2 = random.uniform(64, 128)
        
        new_ind1.append(max(32, min(256, size1)))
        new_ind2.append(max(32, min(256, size2)))
    
    # Blend other parameters
    alpha = random.uniform(0.3, 0.7)
    beta1 = alpha * ind1[num_layers1 + 1] + (1 - alpha) * ind2[num_layers2 + 1]
    beta2 = (1 - alpha) * ind1[num_layers1 + 1] + alpha * ind2[num_layers2 + 1]
    lr1 = alpha * ind1[num_layers1 + 2] + (1 - alpha) * ind2[num_layers2 + 2]
    lr2 = (1 - alpha) * ind1[num_layers1 + 2] + alpha * ind2[num_layers2 + 2]
    dropout1 = alpha * ind1[num_layers1 + 3] + (1 - alpha) * ind2[num_layers2 + 3]
    dropout2 = (1 - alpha) * ind1[num_layers1 + 3] + alpha * ind2[num_layers2 + 3]
    
    new_ind1.extend([beta1, lr1, dropout1])
    new_ind2.extend([beta2, lr2, dropout2])
    
    ind1[:] = new_ind1
    ind2[:] = new_ind2
    
    return ind1, ind2

def evaluate_model_with_depth(individual):
    """Improved evaluation with more training and better metrics"""
    try:
        # Parse individual
        num_layers = int(individual[0])
        layer_sizes = [max(32, min(256, int(round(individual[i])))) 
                       for i in range(1, num_layers + 1)]
        beta = max(0.7, min(0.95, float(individual[num_layers + 1])))
        lr = max(0.001, min(0.005, float(individual[num_layers + 2])))
        dropout = max(0.0, min(0.2, float(individual[num_layers + 3])))
        
        model = create_snn(layer_sizes, beta, dropout)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        # Extended training loop
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
            # Train on more batches based on network depth
            max_batches = min(50, 30 + num_layers * 5)
            if batch_idx >= max_batches:
                break

        # More comprehensive evaluation
        model.eval()
        correct, total = 0, 0
        eval_loss = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = loss_fn(outputs, targets)
                eval_loss += loss.item()
                
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += (pred == targets).sum().item()
                
                # Evaluate on more samples for better accuracy
                if total >= 2000:  # Increased from 1000
                    break
        
        accuracy = correct / total
        avg_train_loss = total_loss / num_batches
        avg_eval_loss = eval_loss / min(batch_idx + 1, len(test_loader))
        
        # Penalize overfitting and complexity
        complexity_penalty = num_layers * 0.001 + sum(layer_sizes) * 0.00001
        overfitting_penalty = max(0, avg_train_loss - avg_eval_loss) * 0.1
        
        final_fitness = accuracy - complexity_penalty - overfitting_penalty
        
        return (final_fitness,)
    
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        return (0.0,)  # Return poor fitness for failed evaluations

def parse_individual(individual):
    """Helper function to parse individual parameters"""
    num_layers = int(individual[0])
    layer_sizes = [int(round(individual[i])) for i in range(1, num_layers + 1)]
    beta = individual[num_layers + 1]
    lr = individual[num_layers + 2]
    dropout = individual[num_layers + 3]
    return num_layers, layer_sizes, beta, lr, dropout

def print_best_individual(generation, best_individual, fitness):
    """Print detailed information about the best individual"""
    num_layers, layer_sizes, beta, lr, dropout = parse_individual(best_individual)
    
    print(f"\n{'='*60}")
    print(f"GENERATION {generation} - BEST INDIVIDUAL")
    print(f"{'='*60}")
    print(f"Fitness: {fitness:.6f}")
    print(f"Architecture: 784 → {' → '.join(map(str, layer_sizes))} → 10")
    print(f"Beta (β): {beta:.4f}")
    print(f"Learning Rate: {lr:.6f}")
    print(f"Dropout Rate: {dropout:.4f}")
    print(f"Total Parameters: ~{sum([784] + layer_sizes + [10]) * sum(layer_sizes + [10]):,}")
    print(f"{'='*60}")

# Create DEAP classes and toolbox
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def create_individual():
    """Create an individual and wrap it in creator.Individual"""
    ind_data = create_individual_with_depth()
    return creator.Individual(ind_data)

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_model_with_depth)
toolbox.register("mate", crossover_with_depth)
toolbox.register("mutate", bounded_mutation_with_depth, mu=0, sigma=0.1, indpb=0.15)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_evolution():
    """Enhanced evolution with detailed best individual tracking"""
    pop_size = 20  # Increased population size
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(3)  # Keep top 3 individuals
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    # Store best individuals from each generation
    generation_best = []

    print("Starting Evolution...")
    print("="*80)

    # Initial evaluation
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    hof.update(pop)
    
    # Find and display best individual of generation 0
    best_ind = tools.selBest(pop, 1)[0]
    generation_best.append((0, best_ind.copy(), best_ind.fitness.values[0]))
    print_best_individual(0, best_ind, best_ind.fitness.values[0])
    
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    print(f"\nGen 0 Stats: {logbook.stream}")
    
    # Evolution loop
    for gen in range(1, 31):  # More generations
        print(f"\n--- Processing Generation {gen} ---")
        
        # Selection and reproduction
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:  # Higher crossover rate
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < 0.2:  # Lower mutation rate
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluation
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Elitism: Keep best individuals
        pop[:] = tools.selBest(offspring + list(hof), len(pop))
        
        hof.update(pop)
        
        # Find and display best individual of current generation
        best_ind = tools.selBest(pop, 1)[0]
        generation_best.append((gen, best_ind.copy(), best_ind.fitness.values[0]))
        print_best_individual(gen, best_ind, best_ind.fitness.values[0])
        
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(f"Gen {gen} Stats: {logbook.stream}")
        
        # Early stopping if no improvement
        if gen > 10 and logbook[-1]['max'] == logbook[-5]['max']:
            print(f"\nEarly stopping at generation {gen} due to no improvement")
            break
    
    # Final summary
    print("\n" + "="*80)
    print("EVOLUTION COMPLETE - FINAL SUMMARY")
    print("="*80)
    
    # Print overall best
    best_overall = max(generation_best, key=lambda x: x[2])
    best_gen, best_ind, best_fitness = best_overall
    
    print(f"\nOVERALL BEST INDIVIDUAL (from Generation {best_gen}):")
    print_best_individual("FINAL", best_ind, best_fitness)
    
    # Print evolution summary
    print(f"\nEvolution Summary:")
    print(f"- Total Generations: {gen}")
    print(f"- Population Size: {pop_size}")
    print(f"- Best Fitness Achieved: {best_fitness:.6f}")
    print(f"- Best Found in Generation: {best_gen}")
    
    return pop, logbook, hof, generation_best

if __name__ == '__main__':
    population, logbook, hall_of_fame, best_individuals = run_evolution()