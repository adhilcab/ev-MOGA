"""
Individual class for the ev-MOGA algorithm.

Represents a single solution in the population with genes and fitness values.
"""

import numpy as np
from typing import List, Any, Optional


class Individual:
    """
    Represents an individual solution in the genetic algorithm population.
    
    An individual contains genes (decision variables) and associated fitness values
    for multiple objectives in the optimization problem.
    """
    
    def __init__(self, genes: Optional[List[float]] = None, num_genes: int = 0):
        """
        Initialize an individual.
        
        Args:
            genes: List of gene values. If None, random genes are generated.
            num_genes: Number of genes if genes is None.
        """
        if genes is not None:
            self.genes = np.array(genes, dtype=float)
        else:
            self.genes = np.random.random(num_genes)
        
        self.fitness_values: List[float] = []
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.dominated_solutions: List['Individual'] = []
        self.domination_count: int = 0
    
    def __len__(self) -> int:
        """Return the number of genes."""
        return len(self.genes)
    
    def __getitem__(self, index: int) -> float:
        """Get gene value at index."""
        return self.genes[index]
    
    def __setitem__(self, index: int, value: float) -> None:
        """Set gene value at index."""
        self.genes[index] = value
    
    def dominates(self, other: 'Individual') -> bool:
        """
        Check if this individual dominates another individual.
        
        An individual A dominates individual B if:
        - A is at least as good as B in all objectives
        - A is strictly better than B in at least one objective
        
        Args:
            other: Another individual to compare with.
            
        Returns:
            True if this individual dominates the other, False otherwise.
        """
        if not self.fitness_values or not other.fitness_values:
            return False
        
        at_least_as_good = all(
            f1 >= f2 for f1, f2 in zip(self.fitness_values, other.fitness_values)
        )
        strictly_better = any(
            f1 > f2 for f1, f2 in zip(self.fitness_values, other.fitness_values)
        )
        
        return at_least_as_good and strictly_better
    
    def copy(self) -> 'Individual':
        """Create a deep copy of the individual."""
        new_individual = Individual(genes=self.genes.copy())
        new_individual.fitness_values = self.fitness_values.copy()
        new_individual.rank = self.rank
        new_individual.crowding_distance = self.crowding_distance
        return new_individual
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """
        Perform mutation on the individual's genes.
        
        Args:
            mutation_rate: Probability of mutating each gene.
            mutation_strength: Strength of the mutation (standard deviation).
        """
        for i in range(len(self.genes)):
            if np.random.random() < mutation_rate:
                self.genes[i] += np.random.normal(0, mutation_strength)
                # Keep genes within [0, 1] bounds
                self.genes[i] = np.clip(self.genes[i], 0.0, 1.0)
    
    def __repr__(self) -> str:
        """String representation of the individual."""
        return f"Individual(genes={self.genes.tolist()}, fitness={self.fitness_values})"