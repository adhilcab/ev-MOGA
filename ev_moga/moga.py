"""
Multi-Objective Genetic Algorithm (MOGA) implementation for Electric Vehicle optimization.

Main algorithm implementation using NSGA-II approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Callable, Dict, Any
from .individual import Individual
from .population import Population
from .objectives import ObjectiveFunction, get_standard_ev_objectives


class MOGA:
    """
    Multi-Objective Genetic Algorithm for Electric Vehicle optimization.
    
    Implements NSGA-II algorithm with customizable objective functions,
    genetic operators, and termination criteria.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        num_genes: int = 8,
        objective_functions: Optional[List[ObjectiveFunction]] = None,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        crossover_rate: float = 0.9,
        tournament_size: int = 3
    ):
        """
        Initialize the MOGA algorithm.
        
        Args:
            population_size: Size of the population.
            num_genes: Number of genes per individual.
            objective_functions: List of objective functions to optimize.
            mutation_rate: Probability of mutation for each gene.
            mutation_strength: Strength of mutation (standard deviation).
            crossover_rate: Probability of crossover.
            tournament_size: Size of tournament for selection.
        """
        self.population_size = population_size
        self.num_genes = num_genes
        self.objective_functions = objective_functions or get_standard_ev_objectives()
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        # Initialize population
        self.population = Population(size=population_size, num_genes=num_genes)
        
        # History tracking
        self.generation = 0
        self.history: Dict[str, List[Any]] = {
            'generations': [],
            'pareto_front_sizes': [],
            'hypervolume': [],
            'average_fitness': []
        }
        
    def evaluate_population(self):
        """Evaluate all individuals in the population using objective functions."""
        objective_funcs = [obj_func for obj_func in self.objective_functions]
        self.population.evaluate(objective_funcs)
    
    def evolve(self, num_generations: int = 100, verbose: bool = True) -> List[Individual]:
        """
        Run the evolutionary algorithm for specified number of generations.
        
        Args:
            num_generations: Number of generations to evolve.
            verbose: Whether to print progress information.
            
        Returns:
            Final Pareto front (list of non-dominated individuals).
        """
        # Initial evaluation
        self.evaluate_population()
        self.population.fast_non_dominated_sort()
        self.population.calculate_crowding_distance()
        
        for generation in range(num_generations):
            self.generation = generation
            
            # Create offspring population
            offspring_population = self._create_offspring()
            
            # Combine parent and offspring populations
            combined_population = Population(size=0, num_genes=self.num_genes)
            combined_population.individuals = (
                self.population.individuals + offspring_population.individuals
            )
            
            # Evaluate offspring
            objective_funcs = [obj_func for obj_func in self.objective_functions]
            combined_population.evaluate(objective_funcs)
            
            # Non-dominated sorting and crowding distance calculation
            combined_population.fast_non_dominated_sort()
            combined_population.calculate_crowding_distance()
            
            # Environmental selection
            self.population = self._environmental_selection(combined_population)
            
            # Record history
            self._record_generation_history()
            
            if verbose and (generation + 1) % 10 == 0:
                pareto_front = self.population.get_pareto_front()
                print(f"Generation {generation + 1}: Pareto front size = {len(pareto_front)}")
        
        return self.get_pareto_front()
    
    def _create_offspring(self) -> Population:
        """Create offspring population through selection, crossover, and mutation."""
        offspring_population = Population(size=0, num_genes=self.num_genes)
        
        while len(offspring_population.individuals) < self.population_size:
            # Selection
            parent1 = self.population.tournament_selection(self.tournament_size)
            parent2 = self.population.tournament_selection(self.tournament_size)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                offspring = self.population.crossover(parent1, parent2)
            else:
                offspring = parent1.copy()
            
            # Mutation
            offspring.mutate(self.mutation_rate, self.mutation_strength)
            
            offspring_population.individuals.append(offspring)
        
        return offspring_population
    
    def _environmental_selection(self, combined_population: Population) -> Population:
        """
        Select individuals for the next generation based on rank and crowding distance.
        
        Args:
            combined_population: Combined parent and offspring population.
            
        Returns:
            Selected population for next generation.
        """
        # Sort by rank and crowding distance
        sorted_individuals = sorted(
            combined_population.individuals,
            key=lambda x: (x.rank, -x.crowding_distance)
        )
        
        # Select top individuals
        new_population = Population(size=0, num_genes=self.num_genes)
        new_population.individuals = sorted_individuals[:self.population_size]
        
        return new_population
    
    def _record_generation_history(self):
        """Record statistics for the current generation."""
        pareto_front = self.population.get_pareto_front()
        
        self.history['generations'].append(self.generation)
        self.history['pareto_front_sizes'].append(len(pareto_front))
        
        # Calculate average fitness for each objective
        if self.population.individuals and self.population.individuals[0].fitness_values:
            avg_fitness = []
            num_objectives = len(self.population.individuals[0].fitness_values)
            
            for obj_idx in range(num_objectives):
                avg_obj_fitness = np.mean([
                    ind.fitness_values[obj_idx] for ind in self.population.individuals
                ])
                avg_fitness.append(avg_obj_fitness)
            
            self.history['average_fitness'].append(avg_fitness)
    
    def get_pareto_front(self) -> List[Individual]:
        """
        Get the current Pareto front.
        
        Returns:
            List of non-dominated individuals.
        """
        return self.population.get_pareto_front()
    
    def plot_pareto_front(self, obj1_idx: int = 0, obj2_idx: int = 1, save_path: Optional[str] = None):
        """
        Plot the Pareto front for two objectives.
        
        Args:
            obj1_idx: Index of first objective to plot.
            obj2_idx: Index of second objective to plot.
            save_path: Optional path to save the plot.
        """
        pareto_front = self.get_pareto_front()
        
        if not pareto_front or not pareto_front[0].fitness_values:
            print("No Pareto front available to plot.")
            return
        
        if len(self.objective_functions) < 2:
            print("Need at least 2 objectives to plot Pareto front.")
            return
        
        # Extract objective values
        obj1_values = [ind.fitness_values[obj1_idx] for ind in pareto_front]
        obj2_values = [ind.fitness_values[obj2_idx] for ind in pareto_front]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(obj1_values, obj2_values, c='red', s=50, alpha=0.7)
        plt.xlabel(f"{self.objective_functions[obj1_idx].name}")
        plt.ylabel(f"{self.objective_functions[obj2_idx].name}")
        plt.title("Pareto Front")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Plot convergence history.
        
        Args:
            save_path: Optional path to save the plot.
        """
        if not self.history['generations']:
            print("No convergence history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot Pareto front size evolution
        ax1.plot(self.history['generations'], self.history['pareto_front_sizes'])
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Pareto Front Size')
        ax1.set_title('Evolution of Pareto Front Size')
        ax1.grid(True, alpha=0.3)
        
        # Plot average fitness evolution
        if self.history['average_fitness']:
            avg_fitness_array = np.array(self.history['average_fitness'])
            for obj_idx in range(avg_fitness_array.shape[1]):
                ax2.plot(
                    self.history['generations'],
                    avg_fitness_array[:, obj_idx],
                    label=f"{self.objective_functions[obj_idx].name}"
                )
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Fitness')
        ax2.set_title('Evolution of Average Fitness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def get_best_individual(self, objective_index: int = 0) -> Individual:
        """
        Get the best individual for a specific objective.
        
        Args:
            objective_index: Index of the objective to optimize.
            
        Returns:
            Best individual for the specified objective.
        """
        if not self.population.individuals:
            raise ValueError("No individuals in population")
        
        return max(
            self.population.individuals,
            key=lambda x: x.fitness_values[objective_index] if x.fitness_values else 0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get algorithm statistics.
        
        Returns:
            Dictionary containing various algorithm statistics.
        """
        pareto_front = self.get_pareto_front()
        
        stats = {
            'generation': self.generation,
            'population_size': len(self.population),
            'pareto_front_size': len(pareto_front),
            'num_objectives': len(self.objective_functions),
            'objective_names': [obj.name for obj in self.objective_functions]
        }
        
        if pareto_front and pareto_front[0].fitness_values:
            # Calculate objective statistics
            fitness_matrix = np.array([ind.fitness_values for ind in pareto_front])
            stats['objective_means'] = fitness_matrix.mean(axis=0).tolist()
            stats['objective_stds'] = fitness_matrix.std(axis=0).tolist()
            stats['objective_ranges'] = (fitness_matrix.max(axis=0) - fitness_matrix.min(axis=0)).tolist()
        
        return stats