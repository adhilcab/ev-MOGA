"""
Objective functions for Electric Vehicle optimization.

Contains various objective functions commonly used in EV optimization problems.
"""

import numpy as np
from typing import Callable, List
from .individual import Individual


class ObjectiveFunction:
    """
    Base class for objective functions in the EV optimization problem.
    """
    
    def __init__(self, name: str, maximize: bool = True):
        """
        Initialize objective function.
        
        Args:
            name: Name of the objective function.
            maximize: Whether to maximize (True) or minimize (False) this objective.
        """
        self.name = name
        self.maximize = maximize
    
    def evaluate(self, individual: Individual) -> float:
        """
        Evaluate the objective function for an individual.
        
        Args:
            individual: Individual to evaluate.
            
        Returns:
            Objective function value.
        """
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def __call__(self, individual: Individual) -> float:
        """Make the objective function callable."""
        value = self.evaluate(individual)
        return value if self.maximize else -value


class BatteryLifeObjective(ObjectiveFunction):
    """
    Objective function for maximizing battery life in electric vehicles.
    
    This objective considers factors like charging patterns, temperature,
    and discharge rates that affect battery longevity.
    """
    
    def __init__(self):
        super().__init__("Battery Life", maximize=True)
    
    def evaluate(self, individual: Individual) -> float:
        """
        Evaluate battery life based on individual's genes.
        
        Genes are expected to represent:
        - genes[0]: Charging rate factor (0-1)
        - genes[1]: Temperature management (0-1)
        - genes[2]: Discharge pattern (0-1)
        - genes[3]: Energy management efficiency (0-1)
        """
        if len(individual.genes) < 4:
            raise ValueError("Individual must have at least 4 genes for battery life evaluation")
        
        charging_rate = individual.genes[0]
        temp_management = individual.genes[1]
        discharge_pattern = individual.genes[2] 
        energy_efficiency = individual.genes[3]
        
        # Battery life is negatively affected by high charging rates
        # and positively affected by good temperature management
        charging_penalty = 1.0 - 0.5 * charging_rate**2
        temp_benefit = temp_management
        discharge_benefit = 1.0 - 0.3 * abs(discharge_pattern - 0.5)
        efficiency_benefit = energy_efficiency
        
        battery_life = (
            charging_penalty * 0.3 +
            temp_benefit * 0.3 +
            discharge_benefit * 0.2 +
            efficiency_benefit * 0.2
        )
        
        return max(0.0, battery_life)


class EnergyEfficiencyObjective(ObjectiveFunction):
    """
    Objective function for maximizing energy efficiency.
    
    Considers factors like motor efficiency, regenerative braking,
    and aerodynamic optimization.
    """
    
    def __init__(self):
        super().__init__("Energy Efficiency", maximize=True)
    
    def evaluate(self, individual: Individual) -> float:
        """
        Evaluate energy efficiency based on individual's genes.
        
        Genes are expected to represent:
        - genes[4]: Motor efficiency (0-1)
        - genes[5]: Regenerative braking (0-1)
        - genes[6]: Aerodynamic factor (0-1)
        - genes[7]: Weight optimization (0-1)
        """
        if len(individual.genes) < 8:
            raise ValueError("Individual must have at least 8 genes for energy efficiency evaluation")
        
        motor_efficiency = individual.genes[4]
        regen_braking = individual.genes[5]
        aerodynamics = individual.genes[6]
        weight_opt = individual.genes[7]
        
        # Efficiency is a weighted combination of factors
        efficiency = (
            motor_efficiency * 0.4 +
            regen_braking * 0.25 +
            aerodynamics * 0.2 +
            weight_opt * 0.15
        )
        
        return efficiency


class RangeObjective(ObjectiveFunction):
    """
    Objective function for maximizing driving range.
    
    Considers battery capacity, energy consumption, and efficiency factors.
    """
    
    def __init__(self):
        super().__init__("Driving Range", maximize=True)
    
    def evaluate(self, individual: Individual) -> float:
        """
        Evaluate driving range based on individual's genes.
        
        Uses energy efficiency and battery parameters to estimate range.
        """
        if len(individual.genes) < 8:
            raise ValueError("Individual must have at least 8 genes for range evaluation")
        
        # Battery capacity factor (normalized)
        battery_capacity = individual.genes[0] * individual.genes[1]
        
        # Energy efficiency from other genes
        energy_efficiency = (
            individual.genes[4] * 0.4 +
            individual.genes[5] * 0.25 +
            individual.genes[6] * 0.2 +
            individual.genes[7] * 0.15
        )
        
        # Range is proportional to battery capacity and efficiency
        range_estimate = battery_capacity * energy_efficiency * 100  # Scale to reasonable range
        
        return range_estimate


class CostObjective(ObjectiveFunction):
    """
    Objective function for minimizing total cost of ownership.
    
    Considers initial cost, maintenance, and energy costs.
    """
    
    def __init__(self):
        super().__init__("Total Cost", maximize=False)  # Minimize cost
    
    def evaluate(self, individual: Individual) -> float:
        """
        Evaluate total cost based on individual's genes.
        
        Lower gene values generally correspond to lower costs.
        """
        if len(individual.genes) < 6:
            raise ValueError("Individual must have at least 6 genes for cost evaluation")
        
        # Higher performance typically means higher cost
        battery_cost = individual.genes[0] * individual.genes[1] * 50000  # Battery cost
        motor_cost = individual.genes[4] * 20000  # Motor cost
        features_cost = sum(individual.genes[2:4]) * 10000  # Additional features
        
        total_cost = battery_cost + motor_cost + features_cost
        
        return total_cost


def get_standard_ev_objectives() -> List[ObjectiveFunction]:
    """
    Get a standard set of objective functions for EV optimization.
    
    Returns:
        List of commonly used objective functions for electric vehicles.
    """
    return [
        BatteryLifeObjective(),
        EnergyEfficiencyObjective(),
        RangeObjective(),
        CostObjective()
    ]


def create_custom_objective(
    name: str, 
    evaluation_func: Callable[[Individual], float], 
    maximize: bool = True
) -> ObjectiveFunction:
    """
    Create a custom objective function.
    
    Args:
        name: Name of the objective function.
        evaluation_func: Function that evaluates an individual.
        maximize: Whether to maximize this objective.
        
    Returns:
        Custom objective function.
    """
    class CustomObjective(ObjectiveFunction):
        def __init__(self):
            super().__init__(name, maximize)
        
        def evaluate(self, individual: Individual) -> float:
            return evaluation_func(individual)
    
    return CustomObjective()