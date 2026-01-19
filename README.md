# ev-MOGA

**Multi-Objective Genetic Algorithm for Electric Vehicle Optimization**

A Python implementation of multi-objective genetic algorithms (MOGA) specifically designed for electric vehicle optimization problems. This library provides a flexible framework for optimizing multiple conflicting objectives such as battery life, energy efficiency, driving range, and cost.

## Features

- **Multi-Objective Optimization**: Simultaneously optimize multiple conflicting objectives
- **NSGA-II Algorithm**: Implementation of the popular Non-dominated Sorting Genetic Algorithm II
- **Electric Vehicle Focus**: Built-in objective functions for common EV optimization problems
- **Extensible Design**: Easy to add custom objective functions and constraints
- **Visualization Tools**: Built-in plotting for Pareto fronts and convergence analysis
- **Well-Documented**: Comprehensive documentation and examples

## Installation

### From Source

```bash
git clone https://github.com/adhilcab/ev-MOGA.git
cd ev-MOGA
pip install -e .
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- SciPy >= 1.7.0

## Quick Start

Here's a simple example to get you started:

```python
from ev_moga import MOGA
from ev_moga.objectives import get_standard_ev_objectives

# Create MOGA instance with standard EV objectives
moga = MOGA(
    population_size=100,
    num_genes=8,
    objective_functions=get_standard_ev_objectives()
)

# Run optimization
pareto_front = moga.evolve(num_generations=50)

# Display results
print(f"Found {len(pareto_front)} solutions in Pareto front")
for solution in pareto_front[:5]:  # Show top 5
    print(f"Objectives: {solution.fitness_values}")
```

## Built-in Objective Functions

The library includes several pre-defined objective functions for electric vehicle optimization:

- **Battery Life**: Maximizes battery longevity considering charging patterns and temperature
- **Energy Efficiency**: Optimizes motor efficiency, regenerative braking, and aerodynamics
- **Driving Range**: Maximizes vehicle range based on battery capacity and efficiency
- **Total Cost**: Minimizes total cost of ownership including initial and operational costs

## Custom Objective Functions

You can easily define custom objective functions:

```python
from ev_moga.objectives import ObjectiveFunction

class AccelerationObjective(ObjectiveFunction):
    def __init__(self):
        super().__init__("Acceleration", maximize=True)
    
    def evaluate(self, individual):
        # Your custom evaluation logic here
        motor_power = individual.genes[0]
        weight_factor = 1.0 - individual.genes[1]
        return motor_power * weight_factor * 100
```

## Examples

Check out the `examples/` directory for comprehensive usage examples:

- `basic_usage.py`: Basic optimization with standard objectives
- `custom_objectives.py`: Creating and using custom objective functions

## Algorithm Parameters

The MOGA class accepts several parameters for customization:

- `population_size`: Number of individuals in the population (default: 100)
- `num_genes`: Number of decision variables per individual (default: 8)
- `mutation_rate`: Probability of gene mutation (default: 0.1)
- `crossover_rate`: Probability of crossover (default: 0.9)
- `tournament_size`: Size of tournament selection (default: 3)

## Visualization

The library provides built-in visualization tools:

```python
# Plot Pareto front
moga.plot_pareto_front(obj1_idx=0, obj2_idx=1)

# Plot convergence history
moga.plot_convergence()
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this library in your research, please cite:

```
@software{ev_moga,
  title={ev-MOGA: Multi-Objective Genetic Algorithm for Electric Vehicle Optimization},
  author={adhilcab},
  url={https://github.com/adhilcab/ev-MOGA},
  year={2024}
}
```
