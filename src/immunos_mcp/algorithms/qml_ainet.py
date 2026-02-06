"""
QML-AiNet: Qualitative Model Learning using Artificial Immune Network

Implementation based on:
- Pang & Coghill (2015) - "QML-AiNet: an immune network approach to learning
  qualitative differential equation models"

Key differences from Opt-AiNet:
1. Modified mutation operator: uniform random selection from constraint space
2. Integer encoding for constraint selection
3. Fitness based on model-data consistency
4. Application to qualitative differential equation learning
"""

from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import sys
from pathlib import Path

# Handle imports for both package use and direct execution
try:
    from .opt_ainet import OptAiNet, Antibody
except ImportError:
    # Direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from immunos_mcp.algorithms.opt_ainet import OptAiNet, Antibody


class QualitativeValue(Enum):
    """Qualitative values for variables."""
    ZERO = "0"
    POSITIVE = "+"
    NEGATIVE = "-"
    LOW = "L"
    MEDIUM = "M"
    HIGH = "H"


class QualitativeChange(Enum):
    """Qualitative changes (derivatives)."""
    STEADY = "std"
    INCREASING = "inc"
    DECREASING = "dec"
    RAPID_INCREASE = "r_inc"
    RAPID_DECREASE = "r_dec"


class QualitativeRelation(Enum):
    """Relations between variables."""
    EQUALS = "="
    PROPORTIONAL_POSITIVE = "∝+"  # M+(X,Y): Y increases when X increases
    PROPORTIONAL_NEGATIVE = "∝-"  # M-(X,Y): Y decreases when X increases
    INFLUENCES_POSITIVE = "I+"     # I+(X,Y): X positively influences Y
    INFLUENCES_NEGATIVE = "I-"     # I-(X,Y): X negatively influences Y


@dataclass
class QualitativeConstraint:
    """
    A qualitative constraint in a QDE model.

    Examples:
    - d(requests)/dt = INCREASING
    - latency = HIGH when requests > threshold
    - d(errors)/dt ∝+ requests  (errors increase with requests)
    """
    variable: str
    relation: QualitativeRelation
    value: Any  # QualitativeValue, QualitativeChange, or another variable
    condition: Optional[str] = None  # Optional condition (e.g., "when X > threshold")

    def __str__(self):
        if self.condition:
            return f"{self.variable} {self.relation.value} {self.value} ({self.condition})"
        return f"{self.variable} {self.relation.value} {self.value}"

    def __repr__(self):
        return self.__str__()


@dataclass
class QDEModel:
    """
    Qualitative Differential Equation Model.

    Represents system behavior using qualitative constraints.
    """
    variables: List[str]
    constraints: List[QualitativeConstraint]
    fitness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        lines = ["QDE Model:"]
        lines.append(f"  Variables: {', '.join(self.variables)}")
        lines.append(f"  Constraints:")
        for c in self.constraints:
            lines.append(f"    - {c}")
        lines.append(f"  Fitness: {self.fitness:.4f}")
        return "\n".join(lines)

    def satisfies(self, observation: Dict[str, Any]) -> bool:
        """
        Check if observation satisfies this QDE model.

        Args:
            observation: Dict mapping variable names to observed values/changes

        Returns:
            True if observation is consistent with model
        """
        # Simplified consistency check
        # In practice, would use qualitative reasoning engine

        for constraint in self.constraints:
            var_value = observation.get(constraint.variable)
            if var_value is None:
                continue

            # Check if constraint is satisfied
            # This is a placeholder - real implementation would use
            # qualitative reasoning algorithms
            if constraint.relation == QualitativeRelation.EQUALS:
                if var_value != constraint.value:
                    return False

        return True


class QMLAiNet(OptAiNet):
    """
    QML-AiNet: Qualitative Model Learning using Immune Network.

    Extends Opt-AiNet with:
    - Integer encoding for constraint selection
    - Modified mutation operator (uniform random selection)
    - Qualitative model representation
    - Model-data consistency evaluation
    """

    def __init__(
        self,
        constraint_space: List[List[QualitativeConstraint]],
        observations: List[Dict[str, Any]],
        variables: List[str],
        population_size: int = 50,
        clone_multiplier: int = 10,
        affinity_threshold: float = 0.1,
        mutation_beta: float = 100.0,
        num_random: int = 20,
        max_generations: int = 100
    ):
        """
        Initialize QML-AiNet.

        Args:
            constraint_space: List of constraint options for each "slot"
                            (position in model). Each slot contains possible
                            constraints for that position.
            observations: Training data - list of observation dicts
            variables: Variable names in the system
            population_size: Initial population size
            clone_multiplier: Clones per antibody
            affinity_threshold: Network suppression threshold
            mutation_beta: Mutation decay parameter
            num_random: Random antibodies per generation
            max_generations: Max iterations
        """
        self.constraint_space = constraint_space
        self.observations = observations
        self.variables = variables

        # Number of dimensions = number of constraint slots
        dimensions = len(constraint_space)

        # Dummy fitness function (overridden)
        def dummy_fitness(x):
            return 0.0

        # Initialize base class (Opt-AiNet)
        # For QML, we use integer positions representing constraint indices
        super().__init__(
            fitness_function=dummy_fitness,
            dimensions=dimensions,
            bounds=(0, 1),  # Will be overridden
            population_size=population_size,
            clone_multiplier=clone_multiplier,
            affinity_threshold=affinity_threshold,
            mutation_beta=mutation_beta,
            num_random=num_random,
            max_generations=max_generations,
            minimize=False  # Maximize fitness (model-data consistency)
        )

    def initialize_population(self):
        """Initialize population with random constraint selections."""
        self.population = []

        for _ in range(self.population_size):
            # Random constraint selection for each slot
            position = np.array([
                np.random.randint(0, len(self.constraint_space[i]))
                for i in range(self.dimensions)
            ], dtype=int)

            antibody = Antibody(position=position)
            antibody.fitness = self._evaluate_qde_model(position)
            self.population.append(antibody)

        self._update_best()

    def _evaluate_qde_model(self, position: np.ndarray) -> float:
        """
        Evaluate QDE model fitness.

        Fitness = consistency with observations.

        Args:
            position: Integer array selecting constraints

        Returns:
            Fitness score (higher = more consistent with data)
        """
        # Decode position to get constraints
        constraints = []
        for i, idx in enumerate(position):
            idx = int(idx) % len(self.constraint_space[i])  # Ensure valid index
            constraints.append(self.constraint_space[i][idx])

        # Create model
        model = QDEModel(
            variables=self.variables,
            constraints=constraints
        )

        # Count how many observations are satisfied
        satisfied_count = 0
        for obs in self.observations:
            if model.satisfies(obs):
                satisfied_count += 1

        # Fitness = fraction of observations explained
        fitness = satisfied_count / len(self.observations) if self.observations else 0.0

        return fitness

    def _mutate(self, position: np.ndarray, mutation_rate: float) -> np.ndarray:
        """
        QML-AiNet modified mutation operator.

        Key difference from Opt-AiNet: Uses UNIFORM RANDOM selection
        from constraint space instead of Gaussian noise.

        This is because adjacent constraints don't have neighborhood
        relationships in qualitative constraint space.

        Args:
            position: Original constraint selection (integer array)
            mutation_rate: Probability of mutating each position

        Returns:
            Mutated position
        """
        mutated = position.copy()

        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Uniform random selection from constraint space
                mutated[i] = np.random.randint(0, len(self.constraint_space[i]))

        return mutated

    def _evaluate(self, position: np.ndarray) -> float:
        """Override base evaluation to use QDE model fitness."""
        return self._evaluate_qde_model(position)

    def decode_model(self, position: np.ndarray) -> QDEModel:
        """
        Decode antibody position to QDE model.

        Args:
            position: Integer array selecting constraints

        Returns:
            QDEModel object
        """
        constraints = []
        for i, idx in enumerate(position):
            idx = int(idx) % len(self.constraint_space[i])
            constraints.append(self.constraint_space[i][idx])

        fitness = self._evaluate_qde_model(position)

        return QDEModel(
            variables=self.variables,
            constraints=constraints,
            fitness=fitness
        )

    def get_best_model(self) -> QDEModel:
        """
        Get the best QDE model found.

        Returns:
            Best QDEModel
        """
        if self.best_antibody is None:
            raise ValueError("No model found. Run optimize() first.")

        return self.decode_model(self.best_antibody.position)

    def get_top_models(self, k: int = 5) -> List[QDEModel]:
        """
        Get top k QDE models.

        Args:
            k: Number of models to return

        Returns:
            List of top QDEModel objects
        """
        sorted_pop = sorted(self.population, key=lambda ab: ab.fitness, reverse=True)

        models = []
        for ab in sorted_pop[:k]:
            model = self.decode_model(ab.position)
            models.append(model)

        return models


# Example: Simple system modeling
if __name__ == '__main__':
    print("=" * 70)
    print("QML-AiNet Example: Learning System Behavior Model")
    print("=" * 70)
    print()

    # Define system variables
    variables = ["requests", "latency", "errors"]

    # Define constraint space
    # For each variable, we can describe its behavior

    constraint_space = [
        # Slot 0: requests behavior
        [
            QualitativeConstraint("requests", QualitativeRelation.EQUALS,
                                QualitativeChange.STEADY),
            QualitativeConstraint("requests", QualitativeRelation.EQUALS,
                                QualitativeChange.INCREASING),
            QualitativeConstraint("requests", QualitativeRelation.EQUALS,
                                QualitativeChange.RAPID_INCREASE),
        ],
        # Slot 1: latency behavior
        [
            QualitativeConstraint("latency", QualitativeRelation.PROPORTIONAL_POSITIVE,
                                "requests"),
            QualitativeConstraint("latency", QualitativeRelation.EQUALS,
                                QualitativeValue.LOW),
            QualitativeConstraint("latency", QualitativeRelation.EQUALS,
                                QualitativeValue.HIGH),
        ],
        # Slot 2: errors behavior
        [
            QualitativeConstraint("errors", QualitativeRelation.PROPORTIONAL_POSITIVE,
                                "latency"),
            QualitativeConstraint("errors", QualitativeRelation.EQUALS,
                                QualitativeValue.ZERO),
            QualitativeConstraint("errors", QualitativeRelation.PROPORTIONAL_POSITIVE,
                                "requests"),
        ]
    ]

    # Training observations (simplified)
    # In real scenario, these would be from actual system monitoring
    observations = [
        {"requests": QualitativeChange.STEADY,
         "latency": QualitativeValue.LOW,
         "errors": QualitativeValue.ZERO},

        {"requests": QualitativeChange.INCREASING,
         "latency": QualitativeValue.MEDIUM,
         "errors": QualitativeValue.LOW},

        {"requests": QualitativeChange.RAPID_INCREASE,
         "latency": QualitativeValue.HIGH,
         "errors": QualitativeValue.MEDIUM},
    ]

    print(f"System Variables: {variables}")
    print(f"Constraint Space Size: {np.prod([len(s) for s in constraint_space])}")
    print(f"Training Observations: {len(observations)}")
    print()

    # Create QML-AiNet learner
    qml = QMLAiNet(
        constraint_space=constraint_space,
        observations=observations,
        variables=variables,
        population_size=30,
        clone_multiplier=8,
        affinity_threshold=0.15,
        max_generations=50
    )

    # Learn model
    print("Learning QDE model...\n")
    best_pos, best_fitness = qml.optimize(verbose=True)

    # Get best model
    print("\n" + "=" * 70)
    print("Best QDE Model Found:")
    print("=" * 70)
    best_model = qml.get_best_model()
    print(best_model)

    # Get alternative models
    print("\n" + "=" * 70)
    print("Top 3 Alternative Models:")
    print("=" * 70)
    top_models = qml.get_top_models(k=3)
    for i, model in enumerate(top_models, 1):
        print(f"\n[Model {i}] Fitness: {model.fitness:.4f}")
        for constraint in model.constraints:
            print(f"  - {constraint}")

    print("\n✓ QML-AiNet successfully learned qualitative models from data!")
