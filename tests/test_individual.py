"""
Unit tests for the Individual class.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ev_moga.individual import Individual


class TestIndividual(unittest.TestCase):
    """Test cases for Individual class."""
    
    def test_initialization_with_genes(self):
        """Test individual initialization with provided genes."""
        genes = [0.1, 0.5, 0.9, 0.3]
        individual = Individual(genes=genes)
        
        self.assertEqual(len(individual), 4)
        self.assertTrue(np.array_equal(individual.genes, np.array(genes)))
        self.assertEqual(individual.fitness_values, [])
        self.assertEqual(individual.rank, 0)
        self.assertEqual(individual.crowding_distance, 0.0)
    
    def test_initialization_random_genes(self):
        """Test individual initialization with random genes."""
        num_genes = 6
        individual = Individual(num_genes=num_genes)
        
        self.assertEqual(len(individual), num_genes)
        self.assertTrue(all(0 <= gene <= 1 for gene in individual.genes))
    
    def test_gene_access(self):
        """Test gene access via indexing."""
        genes = [0.2, 0.7, 0.4]
        individual = Individual(genes=genes)
        
        self.assertEqual(individual[0], 0.2)
        self.assertEqual(individual[1], 0.7)
        self.assertEqual(individual[2], 0.4)
        
        individual[1] = 0.8
        self.assertEqual(individual[1], 0.8)
    
    def test_dominance(self):
        """Test dominance relationship between individuals."""
        # Individual A dominates B if A >= B in all objectives and A > B in at least one
        ind_a = Individual(genes=[0.5, 0.5])
        ind_b = Individual(genes=[0.3, 0.3])
        
        ind_a.fitness_values = [0.8, 0.9]
        ind_b.fitness_values = [0.6, 0.7]
        
        self.assertTrue(ind_a.dominates(ind_b))
        self.assertFalse(ind_b.dominates(ind_a))
    
    def test_no_dominance(self):
        """Test case where neither individual dominates."""
        ind_a = Individual(genes=[0.5, 0.5])
        ind_b = Individual(genes=[0.3, 0.3])
        
        ind_a.fitness_values = [0.8, 0.6]  # Better in first, worse in second
        ind_b.fitness_values = [0.6, 0.8]  # Worse in first, better in second
        
        self.assertFalse(ind_a.dominates(ind_b))
        self.assertFalse(ind_b.dominates(ind_a))
    
    def test_copy(self):
        """Test individual copying."""
        original = Individual(genes=[0.1, 0.2, 0.3])
        original.fitness_values = [0.5, 0.6]
        original.rank = 2
        original.crowding_distance = 1.5
        
        copy = original.copy()
        
        self.assertTrue(np.array_equal(original.genes, copy.genes))
        self.assertEqual(original.fitness_values, copy.fitness_values)
        self.assertEqual(original.rank, copy.rank)
        self.assertEqual(original.crowding_distance, copy.crowding_distance)
        
        # Verify it's a deep copy
        copy.genes[0] = 0.9
        self.assertNotEqual(original.genes[0], copy.genes[0])
    
    def test_mutation(self):
        """Test gene mutation."""
        original_genes = [0.5, 0.5, 0.5, 0.5]
        individual = Individual(genes=original_genes.copy())
        
        # Test with high mutation rate to ensure some genes change
        individual.mutate(mutation_rate=1.0, mutation_strength=0.1)
        
        # Genes should still be within bounds
        self.assertTrue(all(0 <= gene <= 1 for gene in individual.genes))
        
        # At least some genes should have changed (with high probability)
        # Note: This test might occasionally fail due to randomness
        changed = sum(1 for i in range(len(original_genes)) 
                     if abs(individual.genes[i] - original_genes[i]) > 1e-6)
        self.assertGreater(changed, 0)


if __name__ == '__main__':
    unittest.main()