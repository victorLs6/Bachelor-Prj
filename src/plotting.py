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
from matplotlib.patches import Rectangle
from rich.console import Console
from rich.traceback import install

# Global constants
SEED = 42

# Global functions
install(show_locals=True)
console = Console(width=180)
RNG = np.random.default_rng(SEED)
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Warning Control
# Type Checking
# Type Aliases


def parse_individual(individual):
    """Helper function to parse individual parameters."""
    num_layers = int(individual[0])
    layer_sizes = [round(individual[i]) for i in range(1, num_layers + 1)]
    beta = individual[num_layers + 1]
    lr = individual[num_layers + 2]
    dropout = individual[num_layers + 3]
    return num_layers, layer_sizes, beta, lr, dropout


def print_best_individual(generation, best_individual, fitness) -> None:
    """Print detailed information about the best individual."""
    _num_layers, layer_sizes, beta, lr, dropout = parse_individual(
        best_individual,
    )

    console.print(f"\n{'=' * 60}")
    console.print(f"GENERATION {generation} - BEST INDIVIDUAL")
    console.print(f"{'=' * 60}")
    console.print(f"Fitness: {fitness:.6f}")
    console.print(
        f"Architecture: 784 → {' → '.join(map(str, layer_sizes))} → 10",
    )
    console.print(f"Beta (β): {beta:.4f}")
    console.print(f"Learning Rate: {lr:.6f}")
    console.print(f"Dropout Rate: {dropout:.4f}")
    console.print(
        f"Total Parameters: ~{sum([784, *layer_sizes, 10]) * sum([*layer_sizes, 10]):,}",
    )
    console.print(f"{'=' * 60}")


def plot_evolution_progress(logbook, generation_best) -> None:
    """Plot evolution progress with multiple metrics."""
    _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data
    generations = [record["gen"] for record in logbook]
    max_fitness = [record["max"] for record in logbook]
    avg_fitness = [record["avg"] for record in logbook]
    min_fitness = [record["min"] for record in logbook]
    std_fitness = [record["std"] for record in logbook]

    # Plot 1: Fitness Evolution
    ax1.plot(
        generations,
        max_fitness,
        "r-",
        linewidth=2,
        label="Max Fitness",
        marker="o",
    )
    ax1.plot(
        generations,
        avg_fitness,
        "b-",
        linewidth=2,
        label="Avg Fitness",
        marker="s",
    )
    ax1.fill_between(
        generations,
        [avg - std for avg, std in zip(avg_fitness, std_fitness, strict=False)],
        [avg + std for avg, std in zip(avg_fitness, std_fitness, strict=False)],
        alpha=0.3,
        color="blue",
    )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Evolution Progress - Fitness Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best Fitness Trajectory
    best_gens = [item[0] for item in generation_best]
    best_fits = [item[2] for item in generation_best]
    ax2.plot(best_gens, best_fits, "g-", linewidth=3, marker="*", markersize=8)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Best Fitness")
    ax2.set_title("Best Individual Fitness Trajectory")
    ax2.grid(True, alpha=0.3)

    # Annotate improvements
    for i in range(1, len(best_fits)):
        if best_fits[i] > best_fits[i - 1]:
            ax2.annotate(
                f"↑ {best_fits[i]:.4f}",
                xy=(best_gens[i], best_fits[i]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "yellow",
                    "alpha": 0.7,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "connectionstyle": "arc3,rad=0",
                },
            )

    # Plot 3: Fitness Distribution
    try:
        final_gen_data = [
            ind.fitness.values[0]
            for ind in logbook[-1]["population"]
            if hasattr(ind, "fitness") and ind.fitness.valid
        ]
    except KeyError:
        final_gen_data = [max_fitness[-1], avg_fitness[-1], min_fitness[-1]]

    ax3.hist(
        final_gen_data,
        bins=15,
        alpha=0.7,
        color="purple",
        edgecolor="black",
    )
    ax3.axvline(
        np.mean(final_gen_data),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(final_gen_data):.4f}",
    )
    ax3.set_xlabel("Fitness")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Final Generation Fitness Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Convergence Analysis
    improvement_rate = []
    for i in range(1, len(max_fitness)):
        rate = max_fitness[i] - max_fitness[i - 1]
        improvement_rate.append(rate)

    ax4.plot(
        generations[1:],
        improvement_rate,
        "orange",
        linewidth=2,
        marker="d",
    )
    ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Fitness Improvement")
    ax4.set_title("Convergence Rate (Generation-to-Generation Improvement)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/evolution_progress.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_architecture_evolution(generation_best) -> None:
    """Plot architecture evolution over generations."""
    _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract architecture data
    generations = []
    num_layers_list = []
    layer_sizes_by_gen = []
    total_params = []

    for gen, individual, _fitness in generation_best:
        generations.append(gen)
        num_layers, layer_sizes, _beta, _lr, _dropout = parse_individual(
            individual,
        )
        num_layers_list.append(num_layers)
        layer_sizes_by_gen.append(layer_sizes)

        # Calculate approximate parameter count
        params = 784 * layer_sizes[0]  # Input to first hidden
        for i in range(len(layer_sizes) - 1):
            params += layer_sizes[i] * layer_sizes[i + 1]
        params += layer_sizes[-1] * 10  # Last hidden to output
        total_params.append(params)

    # Plot 1: Number of Layers Evolution
    ax1.plot(generations, num_layers_list, "b-o", linewidth=2, markersize=8)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Number of Hidden Layers")
    ax1.set_title("Architecture Depth Evolution")
    ax1.set_ylim(0.5, 10.5)
    ax1.grid(True, alpha=0.3)

    # Annotate layer changes
    for i in range(1, len(num_layers_list)):
        if num_layers_list[i] != num_layers_list[i - 1]:
            change = "+" if num_layers_list[i] > num_layers_list[i - 1] else "-"
            ax1.annotate(
                f"{change}Layer",
                xy=(generations[i], num_layers_list[i]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue"},
                arrowprops={"arrowstyle": "->"},
            )

    # Plot 2: Layer Sizes Heatmap
    max_layers = max(len(sizes) for sizes in layer_sizes_by_gen)
    heatmap_data = np.zeros((len(generations), max_layers))

    for i, sizes in enumerate(layer_sizes_by_gen):
        for j, size in enumerate(sizes):
            heatmap_data[i, j] = size

    im = ax2.imshow(
        heatmap_data.T,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
    )
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Layer Index")
    ax2.set_title("Layer Sizes Evolution Heatmap")
    ax2.set_xticks(range(0, len(generations), max(1, len(generations) // 10)))
    ax2.set_xticklabels(generations[:: max(1, len(generations) // 10)])
    plt.colorbar(im, ax=ax2, label="Layer Size")

    # Plot 3: Total Parameters Evolution
    ax3.plot(generations, total_params, "r-s", linewidth=2, markersize=6)
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Total Parameters")
    ax3.set_title("Model Complexity Evolution")
    ax3.grid(True, alpha=0.3)

    # Format y-axis to show K/M notation
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, p: f"{x / 1000:.0f}K"
            if x < 1000000
            else f"{x / 1000000:.1f}M",
        ),
    )

    # Plot 4: Architecture Visualization for Best Individual
    best_individual = max(generation_best, key=operator.itemgetter(2))
    _, best_layer_sizes, _, _, _ = parse_individual(best_individual[1])

    # Create network diagram
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)

    # Draw layers
    layer_positions = np.linspace(
        1,
        9,
        len(best_layer_sizes) + 2,
    )  # +2 for input and output
    all_sizes = [784, *best_layer_sizes, 10]

    max_neurons = max(all_sizes)
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_sizes)))

    for i, (pos, size, color) in enumerate(
        zip(layer_positions, all_sizes, colors, strict=False),
    ):
        # Scale rectangle height based on layer size
        height = (size / max_neurons) * 6 + 1
        y_center = 5

        rect = Rectangle(
            (pos - 0.3, y_center - height / 2),
            0.6,
            height,
            facecolor=color,
            edgecolor="black",
            alpha=0.7,
        )
        ax4.add_patch(rect)

        # Add labels
        if i == 0:
            label = f"Input\n{size}"
        elif i == len(all_sizes) - 1:
            label = f"Output\n{size}"
        else:
            label = f"H{i}\n{size}"

        ax4.text(
            pos,
            y_center - height / 2 - 0.5,
            label,
            ha="center",
            va="top",
            fontsize=8,
        )

        # Draw connections
        if i < len(layer_positions) - 1:
            ax4.arrow(
                pos + 0.3,
                y_center,
                0.4,
                0,
                head_width=0.1,
                head_length=0.1,
                fc="gray",
                ec="gray",
                alpha=0.5,
            )

    ax4.set_title(
        f"Best Architecture (Gen {best_individual[0]}, Fitness: {best_individual[2]:.4f})",
    )
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["bottom"].set_visible(False)
    ax4.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        "plots/architecture_evolution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_beta_evolution(generation_best) -> None:
    """Plot beta parameter evolution and analysis."""
    _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract parameter data
    generations = []
    betas = []
    learning_rates = []
    dropouts = []
    fitnesses = []

    for gen, individual, fitness in generation_best:
        generations.append(gen)
        _num_layers, _layer_sizes, beta, lr, dropout = parse_individual(
            individual,
        )
        betas.append(beta)
        learning_rates.append(lr)
        dropouts.append(dropout)
        fitnesses.append(fitness)

    # Plot 1: Beta Evolution
    ax1.plot(
        generations,
        betas,
        "purple",
        linewidth=3,
        marker="o",
        markersize=8,
    )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Beta Value")
    ax1.set_title("Beta Parameter Evolution")
    ax1.set_ylim(0.65, 1.0)
    ax1.grid(True, alpha=0.3)

    # Highlight significant changes
    for i in range(1, len(betas)):
        if abs(betas[i] - betas[i - 1]) > 0.05:
            ax1.annotate(
                f"β={betas[i]:.3f}",
                xy=(generations[i], betas[i]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "yellow",
                    "alpha": 0.7,
                },
                arrowprops={"arrowstyle": "->"},
            )

    # Plot 2: Learning Rate Evolution
    ax2.plot(
        generations,
        learning_rates,
        "green",
        linewidth=3,
        marker="s",
        markersize=6,
    )
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Evolution")
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Plot 3: Parameter Correlation with Fitness
    # Create scatter plot with color coding for generations
    scatter = ax3.scatter(
        betas,
        fitnesses,
        c=generations,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black",
    )
    ax3.set_xlabel("Beta Value")
    ax3.set_ylabel("Fitness")
    ax3.set_title("Beta vs Fitness Correlation")
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label="Generation")

    # Add trend line
    z = np.polyfit(betas, fitnesses, 1)
    p = np.poly1d(z)
    ax3.plot(
        betas,
        p(betas),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}",
    )
    ax3.legend()

    # Plot 4: All Parameters Over Time
    # Normalize parameters to [0, 1] for comparison
    epsilon = 1e-10  # Tiny value to prevent division by zero
    norm_betas = [
    (b - min(betas)) / (max(betas) - min(betas) + epsilon) 
    for b in betas
    ]
    norm_lrs = [
        (lr - min(learning_rates)) / (max(learning_rates) - min(learning_rates))
        for lr in learning_rates
    ]
    norm_dropouts = [
        (d - min(dropouts)) / (max(dropouts) - min(dropouts))
        if max(dropouts) > min(dropouts)
        else 0.5
        for d in dropouts
    ]

    ax4.plot(
        generations,
        norm_betas,
        "purple",
        linewidth=2,
        marker="o",
        label="Beta (normalized)",
    )
    ax4.plot(
        generations,
        norm_lrs,
        "green",
        linewidth=2,
        marker="s",
        label="Learning Rate (normalized)",
    )
    ax4.plot(
        generations,
        norm_dropouts,
        "orange",
        linewidth=2,
        marker="^",
        label="Dropout (normalized)",
    )

    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Normalized Parameter Value")
    ax4.set_title("All Parameters Evolution (Normalized)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/beta_evolution.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_layers_evolution(logbook, generation_best) -> None:
    """Plot the evolution of layer counts in the population."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data
    generations = [record["gen"] for record in logbook]
    avg_layers = [record["avg_layers"] for record in logbook]

    # Best individual layers
    best_layers = [ind[1][0] for ind in generation_best]

    # Plot 1: Average layers vs best individual layers
    ax1.plot(
        generations,
        avg_layers,
        "b-",
        linewidth=2,
        marker="o",
        label="Population Average",
    )
    ax1.plot(
        generations,
        best_layers,
        "r-",
        linewidth=2,
        marker="s",
        label="Best Individual",
    )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Number of Layers")
    ax1.set_title("Layer Count Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Layer distribution in final population
    final_pop = logbook[-1]["population"]
    final_layers = [ind[0] for ind in final_pop]

    unique_layers = sorted(set(final_layers))
    counts = [final_layers.count(l) for l in unique_layers]

    ax2.bar(unique_layers, counts, color="purple", alpha=0.7)
    ax2.set_xlabel("Number of Layers")
    ax2.set_ylabel("Count")
    ax2.set_title("Final Population Layer Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/layers_evolution.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_comprehensive_summary_plot(logbook, generation_best) -> None:
    """Create a comprehensive summary plot with key metrics."""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # Extract all data
    generations = [record["gen"] for record in logbook]
    max_fitness = [record["max"] for record in logbook]
    avg_fitness = [record["avg"] for record in logbook]

    gen_data = []
    fitness_data = []
    beta_data = []
    lr_data = []
    layers_data = []
    param_counts = []

    for gen, individual, fitness in generation_best:
        gen_data.append(gen)
        fitness_data.append(fitness)
        num_layers, layer_sizes, beta, lr, _dropout = parse_individual(
            individual,
        )
        beta_data.append(beta)
        lr_data.append(lr)
        layers_data.append(num_layers)

        # Calculate parameters
        params = 784 * layer_sizes[0]
        for i in range(len(layer_sizes) - 1):
            params += layer_sizes[i] * layer_sizes[i + 1]
        params += layer_sizes[-1] * 10
        param_counts.append(params)

    # Main fitness plot (spans 2 columns)
    ax_main = fig.add_subplot(gs[0, :2])
    ax_main.plot(
        generations,
        max_fitness,
        "r-",
        linewidth=3,
        label="Population Max",
        marker="o",
        markersize=6,
    )
    ax_main.plot(
        generations,
        avg_fitness,
        "b-",
        linewidth=2,
        label="Population Avg",
        alpha=0.7,
    )
    ax_main.plot(
        gen_data,
        fitness_data,
        "g-",
        linewidth=3,
        label="Best Individual",
        marker="*",
        markersize=10,
    )
    ax_main.set_xlabel("Generation", fontsize=12)
    ax_main.set_ylabel("Fitness", fontsize=12)
    ax_main.set_title(
        "Evolution Summary - Fitness Progression",
        fontsize=14,
        fontweight="bold",
    )
    ax_main.legend(fontsize=11)
    ax_main.grid(True, alpha=0.3)

    # Best individual info (right column, top)
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis("off")
    best_idx = fitness_data.index(max(fitness_data))
    best_gen = gen_data[best_idx]
    best_fitness = fitness_data[best_idx]
    best_individual = generation_best[best_idx][1]

    _, best_layers, best_beta, best_lr, best_dropout = parse_individual(
        best_individual,
    )

    info_text = f"""BEST INDIVIDUAL

Generation: {best_gen}
Fitness: {best_fitness:.6f}

Architecture:
• Layers: {len(best_layers)}
• Sizes: {best_layers}
• Parameters: ~{param_counts[best_idx]:,}

Parameters:
• Beta: {best_beta:.4f}
• Learning Rate: {best_lr:.6f}
• Dropout: {best_dropout:.4f}"""

    ax_info.text(
        0.05,
        0.95,
        info_text,
        transform=ax_info.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "lightblue",
            "alpha": 0.8,
        },
    )

    # Architecture depth evolution
    ax_depth = fig.add_subplot(gs[1, 0])
    ax_depth.plot(
        gen_data,
        layers_data,
        "purple",
        linewidth=2,
        marker="s",
        markersize=6,
    )
    ax_depth.set_xlabel("Generation")
    ax_depth.set_ylabel("Number of Layers")
    ax_depth.set_title("Architecture Depth")
    ax_depth.grid(True, alpha=0.3)
    ax_depth.set_ylim(1.5, 4.5)

    # Parameter count evolution
    ax_params = fig.add_subplot(gs[1, 1])
    ax_params.plot(
        gen_data,
        param_counts,
        "orange",
        linewidth=2,
        marker="^",
        markersize=6,
    )
    ax_params.set_xlabel("Generation")
    ax_params.set_ylabel("Parameters")
    ax_params.set_title("Model Complexity")
    ax_params.grid(True, alpha=0.3)
    ax_params.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x / 1000:.0f}K"),
    )

    # Beta evolution
    ax_beta = fig.add_subplot(gs[1, 2])
    ax_beta.plot(
        gen_data,
        beta_data,
        "red",
        linewidth=2,
        marker="o",
        markersize=6,
    )
    ax_beta.set_xlabel("Generation")
    ax_beta.set_ylabel("Beta Value")
    ax_beta.set_title("Beta Parameter")
    ax_beta.grid(True, alpha=0.3)
    ax_beta.set_ylim(0.65, 1.0)

    # Performance vs complexity scatter
    ax_scatter = fig.add_subplot(gs[2, 0])
    scatter = ax_scatter.scatter(
        param_counts,
        fitness_data,
        c=gen_data,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black",
    )
    ax_scatter.set_xlabel("Parameter Count")
    ax_scatter.set_ylabel("Fitness")
    ax_scatter.set_title("Performance vs Complexity")
    ax_scatter.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax_scatter, label="Generation")

    # Learning rate evolution
    ax_lr = fig.add_subplot(gs[2, 1])
    ax_lr.plot(
        gen_data,
        lr_data,
        "green",
        linewidth=2,
        marker="d",
        markersize=6,
    )
    ax_lr.set_xlabel("Generation")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_title("Learning Rate Evolution")
    ax_lr.grid(True, alpha=0.3)
    ax_lr.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Improvement rate
    ax_improve = fig.add_subplot(gs[2, 2])
    improvement_rate = [0]  # First generation has no improvement
    improvement_rate.extend(
        fitness_data[i] - fitness_data[i - 1]
        for i in range(1, len(fitness_data))
    )

    colors = ["red" if x < 0 else "green" for x in improvement_rate[1:]]
    ax_improve.bar(gen_data[1:], improvement_rate[1:], color=colors, alpha=0.7)
    ax_improve.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax_improve.set_xlabel("Generation")
    ax_improve.set_ylabel("Fitness Improvement")
    ax_improve.set_title("Generation-to-Generation Progress")
    ax_improve.grid(True, alpha=0.3)

    # Final statistics summary
    ax_stats = fig.add_subplot(gs[3, :])
    ax_stats.axis("off")

    # Calculate statistics
    total_improvement = (
        fitness_data[-1] - fitness_data[0] if len(fitness_data) > 1 else 0
    )
    avg_improvement = (
        total_improvement / len(fitness_data) if len(fitness_data) > 1 else 0
    )
    best_improvement_gen = (
        improvement_rate.index(max(improvement_rate[1:])) + 1
        if len(improvement_rate) > 1
        else 0
    )

    convergence_gen = len(fitness_data)
    for i in range(len(fitness_data) - 5, 0, -1):
        if abs(fitness_data[i] - fitness_data[-1]) > 0.001:
            convergence_gen = i + 5
            break

    stats_text = f"""EVOLUTION STATISTICS

Total Generations: {len(fitness_data)}                    Total Improvement: {total_improvement:.6f}                    Average per Generation: {avg_improvement:.6f}

Best Generation: {best_gen}                    Convergence: ~Gen {convergence_gen}                    Best Improvement: Gen {best_improvement_gen} (+{max(improvement_rate[1:]) if len(improvement_rate) > 1 else 0:.6f})

Architecture Range: {min(layers_data)}-{max(layers_data)} layers          Parameter Range: {min(param_counts):,}-{max(param_counts):,}          Beta Range: {min(beta_data):.3f}-{max(beta_data):.3f}"""

    ax_stats.text(
        0.5,
        0.5,
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        bbox={
            "boxstyle": "round,pad=1",
            "facecolor": "lightyellow",
            "alpha": 0.8,
        },
    )

    plt.suptitle(
        "SNN Evolution Comprehensive Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(
        "plots/comprehensive_evolution_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
