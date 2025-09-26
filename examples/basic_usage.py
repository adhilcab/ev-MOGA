#!/usr/bin/env python3
"""
Basic usage example of the ev-MOGA algorithm.

This example demonstrates how to use the ev-MOGA library to optimize
electric vehicle parameters using multi-objective genetic algorithms.
"""

import sys
import os

# Add the parent directory to Python path to import ev_moga
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ev_moga import MOGA
from ev_moga.objectives import get_standard_ev_objectives, BatteryLifeObjective, EnergyEfficiencyObjective


def main():
    """Run a basic ev-MOGA optimization example."""
    print("=== ev-MOGA Basic Usage Example ===")
    print()
    
    # Create MOGA instance with default EV objectives
    print("Initializing MOGA with standard EV objectives...")
    moga = MOGA(
        population_size=50,  # Smaller population for faster demo
        num_genes=8,         # 8 decision variables
        objective_functions=get_standard_ev_objectives(),
        mutation_rate=0.1,
        crossover_rate=0.9
    )
    
    print(f"Population size: {moga.population_size}")
    print(f"Number of genes per individual: {moga.num_genes}")
    print(f"Objectives: {[obj.name for obj in moga.objective_functions]}")
    print()
    
    # Run optimization
    print("Running optimization for 20 generations...")
    pareto_front = moga.evolve(num_generations=20, verbose=True)
    
    print()
    print(f"Final Pareto front contains {len(pareto_front)} solutions")
    
    # Display some results
    print("\n=== Top 5 Solutions from Pareto Front ===")
    for i, individual in enumerate(pareto_front[:5]):
        print(f"Solution {i+1}:")
        print(f"  Genes: {[f'{gene:.3f}' for gene in individual.genes]}")
        print(f"  Objectives: {[f'{obj:.3f}' for obj in individual.fitness_values]}")
        print()
    
    # Get best individual for each objective
    print("=== Best Individual for Each Objective ===")
    for obj_idx, obj_func in enumerate(moga.objective_functions):
        best_ind = moga.get_best_individual(obj_idx)
        print(f"{obj_func.name}: {best_ind.fitness_values[obj_idx]:.3f}")
    
    # Show algorithm statistics
    print("\n=== Algorithm Statistics ===")
    stats = moga.get_statistics()
    for key, value in stats.items():
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], float):
                print(f"{key}: {[f'{v:.3f}' for v in value]}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    
    # Plot results (optional - will show plots if running interactively)
    try:
        print("\n=== Generating Plots ===")
        print("Plotting Pareto front...")
        moga.plot_pareto_front(obj1_idx=0, obj2_idx=1, save_path="pareto_front.png")
        
        print("Plotting convergence history...")
        moga.plot_convergence(save_path="convergence.png")
        
        print("Plots saved as 'pareto_front.png' and 'convergence.png'")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()