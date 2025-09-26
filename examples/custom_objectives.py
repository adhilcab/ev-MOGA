#!/usr/bin/env python3
"""
Example showing how to create and use custom objective functions.

This example demonstrates how to define custom optimization objectives
for specific electric vehicle use cases.
"""

import sys
import os
import numpy as np

# Add the parent directory to Python path to import ev_moga
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ev_moga import MOGA, Individual
from ev_moga.objectives import ObjectiveFunction, create_custom_objective


class AccelerationObjective(ObjectiveFunction):
    """Custom objective for maximizing acceleration performance."""
    
    def __init__(self):
        super().__init__("Acceleration", maximize=True)
    
    def evaluate(self, individual: Individual) -> float:
        """Evaluate acceleration based on motor power and weight."""
        if len(individual.genes) < 4:
            return 0.0
        
        motor_power = individual.genes[0]  # Motor power factor
        weight_factor = 1.0 - individual.genes[1]  # Lower weight = better acceleration
        torque_factor = individual.genes[2]  # Torque delivery
        efficiency = individual.genes[3]  # Power efficiency
        
        # Acceleration score (0-100 scale)
        acceleration_score = (
            motor_power * 0.4 +
            weight_factor * 0.3 +
            torque_factor * 0.2 +
            efficiency * 0.1
        ) * 100
        
        return acceleration_score


class NoiseObjective(ObjectiveFunction):
    """Custom objective for minimizing vehicle noise."""
    
    def __init__(self):
        super().__init__("Noise Level", maximize=False)  # Minimize noise
    
    def evaluate(self, individual: Individual) -> float:
        """Evaluate noise level based on motor and aerodynamic factors."""
        if len(individual.genes) < 6:
            return 100.0  # High noise penalty
        
        motor_noise = individual.genes[0] * 30  # Motor contribution to noise
        aero_noise = individual.genes[4] * 20   # Aerodynamic noise
        tire_noise = individual.genes[5] * 15   # Tire noise
        insulation = (1.0 - individual.genes[1]) * 10  # Insulation reduces noise
        
        total_noise = motor_noise + aero_noise + tire_noise + insulation
        return total_noise


def simple_range_objective(individual: Individual) -> float:
    """Simple range calculation for demonstration."""
    battery_capacity = individual.genes[0] if len(individual.genes) > 0 else 0.5
    efficiency = individual.genes[1] if len(individual.genes) > 1 else 0.5
    return battery_capacity * efficiency * 200  # Range in km


def main():
    """Run custom objectives optimization example."""
    print("=== ev-MOGA Custom Objectives Example ===")
    print()
    
    # Create custom objective functions
    acceleration_obj = AccelerationObjective()
    noise_obj = NoiseObjective()
    range_obj = create_custom_objective("Simple Range", simple_range_objective, maximize=True)
    
    custom_objectives = [acceleration_obj, noise_obj, range_obj]
    
    print("Custom objectives created:")
    for obj in custom_objectives:
        print(f"  - {obj.name} ({'maximize' if obj.maximize else 'minimize'})")
    print()
    
    # Create MOGA with custom objectives
    moga = MOGA(
        population_size=30,
        num_genes=6,  # Fewer genes for this example
        objective_functions=custom_objectives,
        mutation_rate=0.15,
        crossover_rate=0.8
    )
    
    print("Running optimization with custom objectives...")
    pareto_front = moga.evolve(num_generations=15, verbose=True)
    
    print(f"\nOptimization completed! Pareto front size: {len(pareto_front)}")
    
    # Analyze results
    print("\n=== Analysis of Custom Objectives ===")
    
    # Find trade-offs between objectives
    print("\nTrade-off analysis:")
    for individual in pareto_front[:3]:  # Show top 3 solutions
        print(f"Solution: genes = {[f'{g:.3f}' for g in individual.genes]}")
        for obj_idx, obj in enumerate(custom_objectives):
            value = individual.fitness_values[obj_idx]
            print(f"  {obj.name}: {value:.2f}")
        print()
    
    # Show objective correlations
    if len(pareto_front) > 1:
        print("Objective value ranges in Pareto front:")
        fitness_matrix = np.array([ind.fitness_values for ind in pareto_front])
        
        for obj_idx, obj in enumerate(custom_objectives):
            obj_values = fitness_matrix[:, obj_idx]
            print(f"  {obj.name}: {obj_values.min():.2f} - {obj_values.max():.2f}")
    
    # Demonstrate individual objective evaluation
    print("\n=== Individual Objective Evaluation Demo ===")
    test_individual = Individual(genes=[0.8, 0.3, 0.9, 0.7, 0.5, 0.4])
    print(f"Test individual genes: {test_individual.genes.tolist()}")
    
    for obj in custom_objectives:
        value = obj.evaluate(test_individual)
        print(f"  {obj.name}: {value:.2f}")
    
    print("\n=== Custom objectives example completed! ===")


if __name__ == "__main__":
    main()