"""
Opt-AiNet: Optimization Artificial Immune Network

Implementation based on:
- de Castro & Timmis (2002) - "An artificial immune network for multimodal function optimization"
- Clever Algorithms documentation

Core principles:
- Network suppression for diversity maintenance
- Clonal selection with fitness-based mutation
- Multi-modal optimization through network interactions
"""

from typing import List, Callable, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import time


@dataclass
class Antibody:
    """
    Antibody (candidate solution) in the immune network.

    Attributes:
        position: Solution vector in search space
        fitness: Fitness score (higher is better)
        age: Generation count since creation
        clone_count: Number of clones to generate
    """
    position: np.ndarray
    fitness: float = 0.0
    age: int = 0
    clone_count: int = 0

    def __hash__(self):
        return hash(tuple(self.position))


class OptAiNet:
    """
    Optimization Artificial Immune Network (Opt-AiNet).

    Multi-modal optimization using immune network principles:
    - Clonal selection for exploitation
    - Network suppression for diversity
    - Random insertion for exploration
    """

    def __init__(
        self,
        fitness_function: Callable[[np.ndarray], float],
        dimensions: int,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        population_size: int = 50,
        clone_multiplier: int = 10,
        affinity_threshold: float = 0.05,
        mutation_beta: float = 100.0,
        num_random: int = 20,
        max_generations: int = 100,
        minimize: bool = False
    ):
        """
        Initialize Opt-AiNet.

        Args:
            fitness_function: Function to maximize (or minimize if minimize=True)
            dimensions: Dimensionality of search space
            bounds: (min, max) bounds for each dimension
            population_size: Initial population size
            clone_multiplier: Number of clones per antibody
            affinity_threshold: Suppression threshold (as fraction of search space)
            mutation_beta: Controls mutation decay (higher = less mutation)
            num_random: Number of random antibodies to insert each generation
            max_generations: Maximum iterations
            minimize: If True, minimizes fitness; if False, maximizes
        """
        self.fitness_function = fitness_function
        self.dimensions = dimensions
        self.bounds = bounds
        self.population_size = population_size
        self.clone_multiplier = clone_multiplier
        self.affinity_threshold = affinity_threshold
        self.mutation_beta = mutation_beta
        self.num_random = num_random
        self.max_generations = max_generations
        self.minimize = minimize

        # State
        self.population: List[Antibody] = []
        self.generation = 0
        self.best_antibody: Optional[Antibody] = None
        self.history = []

        # Calculate absolute affinity threshold
        search_space_size = bounds[1] - bounds[0]
        self.affinity_threshold_abs = affinity_threshold * search_space_size

    def initialize_population(self):
        """Initialize population with random antibodies."""
        self.population = []
        for _ in range(self.population_size):
            position = np.random.uniform(
                self.bounds[0],
                self.bounds[1],
                size=self.dimensions
            )
            antibody = Antibody(position=position)
            antibody.fitness = self._evaluate(antibody.position)
            self.population.append(antibody)

        self._update_best()

    def _evaluate(self, position: np.ndarray) -> float:
        """Evaluate fitness (handle minimization)."""
        fitness = self.fitness_function(position)
        # For minimization, negate fitness so higher is still better internally
        return -fitness if self.minimize else fitness

    def _update_best(self):
        """Update best antibody found so far."""
        if not self.population:
            return

        current_best = max(self.population, key=lambda ab: ab.fitness)

        if self.best_antibody is None or current_best.fitness > self.best_antibody.fitness:
            self.best_antibody = Antibody(
                position=current_best.position.copy(),
                fitness=current_best.fitness,
                age=current_best.age
            )

    def clone_and_mutate(self, antibody: Antibody) -> List[Antibody]:
        """
        Clone antibody and mutate clones.

        Better antibodies (higher fitness) are:
        - Cloned more
        - Mutated less

        Args:
            antibody: Parent antibody to clone

        Returns:
            List of mutated clones
        """
        # Number of clones proportional to fitness
        num_clones = int(self.clone_multiplier * antibody.fitness /
                        (max(ab.fitness for ab in self.population) + 1e-10))
        num_clones = max(1, min(num_clones, self.clone_multiplier))

        clones = []
        for _ in range(num_clones):
            # Mutation rate inversely proportional to fitness
            mutation_rate = np.exp(-self.mutation_beta * antibody.fitness /
                                   (max(ab.fitness for ab in self.population) + 1e-10))

            # Mutate position
            mutated_position = self._mutate(antibody.position, mutation_rate)

            # Create clone
            clone = Antibody(position=mutated_position)
            clone.fitness = self._evaluate(clone.position)
            clones.append(clone)

        return clones

    def _mutate(self, position: np.ndarray, mutation_rate: float) -> np.ndarray:
        """
        Mutate position with Gaussian noise.

        Args:
            position: Original position
            mutation_rate: Mutation strength (0-1)

        Returns:
            Mutated position (clamped to bounds)
        """
        noise = np.random.randn(self.dimensions) * mutation_rate
        mutated = position + noise

        # Clamp to bounds
        mutated = np.clip(mutated, self.bounds[0], self.bounds[1])

        return mutated

    def network_suppression(self, population: List[Antibody]) -> List[Antibody]:
        """
        Network suppression: eliminate similar, lower-fitness antibodies.

        This maintains diversity by preventing redundant solutions.

        Args:
            population: Current population

        Returns:
            Suppressed population (without redundant antibodies)
        """
        if len(population) <= 1:
            return population

        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda ab: ab.fitness, reverse=True)

        suppressed = []
        for i, antibody in enumerate(sorted_pop):
            # Check if suppressed by any higher-fitness antibody already kept
            is_suppressed = False

            for kept in suppressed:
                distance = np.linalg.norm(antibody.position - kept.position)

                if distance < self.affinity_threshold_abs:
                    # Too similar to a better antibody - suppress
                    is_suppressed = True
                    break

            if not is_suppressed:
                suppressed.append(antibody)

        return suppressed

    def add_random_antibodies(self, num_random: int) -> List[Antibody]:
        """
        Add random antibodies for exploration.

        Args:
            num_random: Number of random antibodies to generate

        Returns:
            List of random antibodies
        """
        random_abs = []
        for _ in range(num_random):
            position = np.random.uniform(
                self.bounds[0],
                self.bounds[1],
                size=self.dimensions
            )
            antibody = Antibody(position=position)
            antibody.fitness = self._evaluate(antibody.position)
            random_abs.append(antibody)

        return random_abs

    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run Opt-AiNet optimization.

        Args:
            verbose: Print progress

        Returns:
            Tuple of (best_position, best_fitness)
        """
        start_time = time.time()

        # Initialize
        self.initialize_population()

        if verbose:
            print(f"Opt-AiNet Optimization")
            print(f"  Dimensions: {self.dimensions}")
            print(f"  Population: {self.population_size}")
            print(f"  Generations: {self.max_generations}")
            print(f"  Affinity threshold: {self.affinity_threshold}")
            print()

        for gen in range(self.max_generations):
            self.generation = gen

            # 1. Clone and mutate
            all_clones = []
            for antibody in self.population:
                clones = self.clone_and_mutate(antibody)
                all_clones.extend(clones)

            # 2. Combine population with clones
            combined = self.population + all_clones

            # 3. Network suppression
            self.population = self.network_suppression(combined)

            # 4. Add random antibodies for exploration
            random_abs = self.add_random_antibodies(self.num_random)
            self.population.extend(random_abs)

            # 5. Update best
            self._update_best()

            # 6. Age antibodies
            for ab in self.population:
                ab.age += 1

            # Track history
            best_fitness = self.best_antibody.fitness if self.best_antibody else 0
            avg_fitness = np.mean([ab.fitness for ab in self.population])
            self.history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'population_size': len(self.population)
            })

            if verbose and (gen % 10 == 0 or gen == self.max_generations - 1):
                # Convert back if minimizing
                display_fitness = -best_fitness if self.minimize else best_fitness
                print(f"Gen {gen:3d}: Best={display_fitness:.6f}, "
                      f"Avg={avg_fitness:.6f}, Pop={len(self.population)}")

        elapsed = time.time() - start_time

        if verbose:
            print(f"\nOptimization complete in {elapsed:.2f}s")
            final_fitness = -self.best_antibody.fitness if self.minimize else self.best_antibody.fitness
            print(f"Best fitness: {final_fitness:.6f}")
            print(f"Final population: {len(self.population)} antibodies")

        # Return best solution (convert fitness back if minimizing)
        best_fitness = -self.best_antibody.fitness if self.minimize else self.best_antibody.fitness
        return self.best_antibody.position, best_fitness

    def get_all_optima(self, top_k: int = 10) -> List[Tuple[np.ndarray, float]]:
        """
        Get multiple optimal solutions (for multi-modal optimization).

        Args:
            top_k: Number of top solutions to return

        Returns:
            List of (position, fitness) tuples
        """
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda ab: ab.fitness, reverse=True)

        results = []
        for ab in sorted_pop[:top_k]:
            fitness = -ab.fitness if self.minimize else ab.fitness
            results.append((ab.position.copy(), fitness))

        return results


# Example usage and test functions
def sphere_function(x: np.ndarray) -> float:
    """Simple unimodal test function: f(x) = sum(x^2)"""
    return np.sum(x ** 2)


def rastrigin_function(x: np.ndarray) -> float:
    """Multi-modal test function with many local optima."""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def ackley_function(x: np.ndarray) -> float:
    """Multi-modal test function."""
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))

    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


if __name__ == '__main__':
    print("=" * 70)
    print("Opt-AiNet Test: Sphere Function (Unimodal)")
    print("=" * 70)

    optimizer = OptAiNet(
        fitness_function=sphere_function,
        dimensions=5,
        bounds=(-5.0, 5.0),
        population_size=30,
        clone_multiplier=10,
        affinity_threshold=0.1,
        max_generations=50,
        minimize=True  # Minimize sphere function
    )

    best_pos, best_fit = optimizer.optimize(verbose=True)
    print(f"\nBest position: {best_pos}")
    print(f"Best fitness: {best_fit:.8f}")
    print(f"Expected: 0.0 at origin")

    print("\n" + "=" * 70)
    print("Opt-AiNet Test: Rastrigin Function (Multi-modal)")
    print("=" * 70)

    optimizer = OptAiNet(
        fitness_function=rastrigin_function,
        dimensions=2,
        bounds=(-5.12, 5.12),
        population_size=50,
        clone_multiplier=10,
        affinity_threshold=0.05,
        max_generations=100,
        minimize=True
    )

    best_pos, best_fit = optimizer.optimize(verbose=True)

    # Get multiple optima
    print("\nTop 5 solutions found:")
    optima = optimizer.get_all_optima(top_k=5)
    for i, (pos, fit) in enumerate(optima, 1):
        print(f"  {i}. Position: {pos}, Fitness: {fit:.6f}")
