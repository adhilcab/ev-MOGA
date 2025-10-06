"""
Population class for the ev-MOGA algorithm.

Manages a collection of individuals and provides population-level operations.
"""

import numpy as np
from typing import List, Callable
from .individual import Individual


class Population:
    """
    Represents a population of individuals in the genetic algorithm.
    
    Provides methods for population initialization, evaluation, selection,
    and other genetic operations.
    """
    
    def __init__(self, size: int = 100, num_genes: int = 10):
        """
        Initialize a population.
        
        Args:
            size: Number of individuals in the population.
            num_genes: Number of genes per individual.
        """
        self.size = size
        self.num_genes = num_genes
        self.individuals: List[Individual] = []
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize the population with random individuals."""
        self.individuals = [
            Individual(num_genes=self.num_genes) for _ in range(self.size)
        ]
    
    def evaluate(self, objective_functions: List[Callable[[Individual], float]]):
        """
        Evaluate all individuals in the population using given objective functions.
        
        Args:
            objective_functions: List of functions that evaluate an individual.
        """
        for individual in self.individuals:
            individual.fitness_values = [
                obj_func(individual) for obj_func in objective_functions
            ]
    
    def fast_non_dominated_sort(self):
        """
        Perform fast non-dominated sorting on the population.
        
        Assigns rank to each individual based on Pareto dominance.
        """
        # Initialize domination relationships
        for individual in self.individuals:
            individual.dominated_solutions = []
            individual.domination_count = 0
        
        # Calculate domination relationships
        for i, individual_i in enumerate(self.individuals):
            for j, individual_j in enumerate(self.individuals):
                if i != j:
                    if individual_i.dominates(individual_j):
                        individual_i.dominated_solutions.append(individual_j)
                    elif individual_j.dominates(individual_i):
                        individual_i.domination_count += 1
        
        # Assign ranks
        current_front = []
        for individual in self.individuals:
            if individual.domination_count == 0:
                individual.rank = 1
                current_front.append(individual)
        
        rank = 1
        while current_front:
            next_front = []
            for individual in current_front:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = rank + 1
                        next_front.append(dominated)
            current_front = next_front
            rank += 1
    
    def calculate_crowding_distance(self):
        """Calculate crowding distance for all individuals."""
        if not self.individuals or not self.individuals[0].fitness_values:
            return
        
        num_objectives = len(self.individuals[0].fitness_values)
        
        # Initialize crowding distance
        for individual in self.individuals:
            individual.crowding_distance = 0.0
        
        # Calculate crowding distance for each objective
        for obj_index in range(num_objectives):
            # Sort individuals by objective value
            sorted_individuals = sorted(
                self.individuals, 
                key=lambda x: x.fitness_values[obj_index]
            )
            
            # Set boundary individuals to infinite distance
            sorted_individuals[0].crowding_distance = float('inf')
            sorted_individuals[-1].crowding_distance = float('inf')
            
            # Calculate distance for interior individuals
            obj_range = (
                sorted_individuals[-1].fitness_values[obj_index] -
                sorted_individuals[0].fitness_values[obj_index]
            )
            
            if obj_range > 0:
                for i in range(1, len(sorted_individuals) - 1):
                    distance = (
                        sorted_individuals[i + 1].fitness_values[obj_index] -
                        sorted_individuals[i - 1].fitness_values[obj_index]
                    ) / obj_range
                    sorted_individuals[i].crowding_distance += distance
    
    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """
        Perform tournament selection to choose an individual.
        
        Args:
            tournament_size: Number of individuals in each tournament.
            
        Returns:
            Selected individual.
        """
        # Select random indices for tournament
        tournament_indices = np.random.choice(
            len(self.individuals), size=tournament_size, replace=False
        )
        tournament = [self.individuals[i] for i in tournament_indices]
        
        # Select based on rank and crowding distance
        best = tournament[0]
        for individual in tournament[1:]:
            if (individual.rank < best.rank or 
                (individual.rank == best.rank and 
                 individual.crowding_distance > best.crowding_distance)):
                best = individual
        
        return best
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Perform crossover between two parents to create offspring.
        
        Args:
            parent1: First parent individual.
            parent2: Second parent individual.
            
        Returns:
            Offspring individual.
        """
        # Simulated Binary Crossover (SBX)
        eta_c = 20  # Distribution index for crossover
        offspring_genes = np.zeros(len(parent1.genes))
        
        for i in range(len(parent1.genes)):
            if np.random.random() <= 0.5:  # Crossover probability
                if abs(parent1.genes[i] - parent2.genes[i]) > 1e-14:
                    if parent1.genes[i] < parent2.genes[i]:
                        y1, y2 = parent1.genes[i], parent2.genes[i]
                    else:
                        y1, y2 = parent2.genes[i], parent1.genes[i]
                    
                    rand = np.random.random()
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta_c + 1))
                    
                    offspring_genes[i] = 0.5 * ((y1 + y2) - beta * abs(y2 - y1))
                else:
                    offspring_genes[i] = parent1.genes[i]
            else:
                offspring_genes[i] = parent1.genes[i]
        
        # Ensure genes are within bounds
        offspring_genes = np.clip(offspring_genes, 0.0, 1.0)
        return Individual(genes=offspring_genes.tolist())
    
    def get_pareto_front(self) -> List[Individual]:
        """
        Get individuals in the first Pareto front (rank 1).
        
        Returns:
            List of individuals in the first Pareto front.
        """
        return [ind for ind in self.individuals if ind.rank == 1]
    
    def __len__(self) -> int:
        """Return the size of the population."""
        return len(self.individuals)
    
    def __getitem__(self, index: int) -> Individual:
        """Get individual at index."""
        return self.individuals[index]