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
import operator
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from deap import base, creator, tools
from rich.console import Console
from rich.traceback import install

# Local libraries
from genotype import (
    bounded_mutation_with_depth,
    create_individual_with_depth,
    crossover_with_depth,
    evaluate_model_with_depth,
)
from plotting import (
    create_comprehensive_summary_plot,
    plot_architecture_evolution,
    plot_beta_evolution,
    plot_evolution_progress,
    plot_layers_evolution,
    print_best_individual,
)

# Global constants
SEED = 42
POP_SIZE = 20

# Global functions
install(show_locals=True)
console = Console(width=180)
RNG = np.random.default_rng(SEED)
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Warning Control
# Type Checking
# Type Aliases

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def create_individual():
    """Create an individual and wrap it in creator.Individual."""
    ind_data = create_individual_with_depth()
    return creator.Individual(ind_data)


toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_model_with_depth)
toolbox.register("mate", crossover_with_depth)
toolbox.register(
    "mutate",
    bounded_mutation_with_depth,
    mu=0,
    sigma=0.1,
    indpb=0.15,
)
toolbox.register("select", tools.selTournament, tournsize=POP_SIZE // 4)


def run_evolution():
    pop_size = POP_SIZE
    pop = toolbox.population(POP_SIZE)
    hof = tools.HallOfFame(POP_SIZE // 4)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)

    # Add a new statistic for average layers
    stats_layers = tools.Statistics(
        operator.itemgetter(0),
    )  # ind[0] is the number of layers
    stats_layers.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", *stats.fields]

    # Store best individuals from each generation
    generation_best = []
    # Store populations for each generation
    populations = []
    # Store average layers per generation
    avg_layers_per_gen = []

    console.print("Starting Evolution...")
    console.print("=" * 80)

    # Initial evaluation
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses, strict=False):
        ind.fitness.values = fit

    hof.update(pop)
    populations.append(pop)  # Store initial population

    # Calculate and store average layers for generation 0
    avg_layers = stats_layers.compile(pop)["avg"]
    avg_layers_per_gen.append(avg_layers)

    # Find and display best individual of generation 0
    best_ind = tools.selBest(pop, 1)[0]
    generation_best.append((0, best_ind.copy(), best_ind.fitness.values[0]))
    print_best_individual(0, best_ind, best_ind.fitness.values[0])

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    console.print(f"\nGen 0 Stats:\n {logbook.stream}")
    console.print(f"Average layers in population: {avg_layers:.2f}")

    # Evolution loop
    for gen in range(1, 31):
        console.print(f"\n--- Processing Generation {gen} ---")

        # Selection and reproduction
        offspring = tools.selTournament(
            individuals=pop,
            k=POP_SIZE // 2,
            tournsize=POP_SIZE // 4,
        )
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(
            offspring[::2],
            offspring[1::2],
            strict=False,
        ):
            if RNG.random() < 0.6:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluation
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses, strict=False):
            ind.fitness.values = fit

        # Elitism: Keep best individuals
        pop[:] = tools.selTournament(
            individuals=pop + offspring,
            k=POP_SIZE,
            tournsize=POP_SIZE // 4,
        )

        hof.update(pop)
        populations.append(pop.copy())  # Store current population

        # Calculate and store average layers for this generation
        avg_layers = stats_layers.compile(pop)["avg"]
        avg_layers_per_gen.append(avg_layers)

        # Find and display best individual of current generation
        best_ind = tools.selBest(pop, 1)[0]
        generation_best.append((
            gen,
            best_ind.copy(),
            best_ind.fitness.values[0],
        ))
        print_best_individual(gen, best_ind, best_ind.fitness.values[0])

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        console.print(f"Gen {gen} \nStats: {logbook.stream}")
        console.print(f"Average layers in population: {avg_layers:.2f}")

        # Early stopping if no improvement
        if gen > 10 and logbook[-1]["max"] == logbook[-5]["max"]:
            console.print(
                f"\nEarly stopping at generation {gen} due to no improvement",
            )
            break

    # Add population information to logbook
    for i, entry in enumerate(logbook):
        entry["population"] = populations[i]
        entry["avg_layers"] = avg_layers_per_gen[
            i
        ]  # Add average layers to logbook

    # Plot the average layers evolution
    plot_layers_evolution(logbook, generation_best)

    # Final summary
    console.print("\n" + "=" * 80)
    console.print("EVOLUTION COMPLETE - FINAL SUMMARY")
    console.print("=" * 80)

    # Print overall best
    best_overall = max(generation_best, key=operator.itemgetter(2))
    best_gen, best_ind, best_fitness = best_overall

    console.print(f"\nOVERALL BEST INDIVIDUAL (from Generation {best_gen}):")
    print_best_individual("FINAL", best_ind, best_fitness)

    # Print evolution summary
    console.print("\nEvolution Summary:")
    console.print(f"- Total Generations: {gen}")
    console.print(f"- Population Size: {pop_size}")
    console.print(f"- Best Fitness Achieved: {best_fitness:.6f}")
    console.print(f"- Best Found in Generation: {best_gen}")

    # Generate all plots
    console.print("\nGenerating plots...")

    console.print("1. Evolution Progress Plot...")
    plot_evolution_progress(logbook, generation_best)

    console.print("2. Architecture Evolution Plot...")
    plot_architecture_evolution(generation_best)

    console.print("3. Beta Evolution Plot...")
    plot_beta_evolution(generation_best)

    console.print("4. Comprehensive Summary Plot...")
    create_comprehensive_summary_plot(logbook, generation_best)

    console.print("All plots saved successfully!")

    return pop, logbook, hof, generation_best


if __name__ == "__main__":
    population, logbook, hall_of_fame, best_individuals = run_evolution()
