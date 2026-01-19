"""
ev-MOGA: Multi-Objective Genetic Algorithm for Electric Vehicle optimization

A Python implementation of multi-objective genetic algorithms specifically
designed for electric vehicle optimization problems.
"""

__version__ = "0.1.0"
__author__ = "adhilcab"
__description__ = "Multi-Objective Genetic Algorithm for Electric Vehicle optimization"

from .moga import MOGA
from .individual import Individual
from .population import Population
from .objectives import ObjectiveFunction

__all__ = ["MOGA", "Individual", "Population", "ObjectiveFunction"]