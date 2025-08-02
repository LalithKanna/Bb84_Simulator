"""
Unit tests for quantum operations
"""
import unittest
import numpy as np
from modules.polarization.quantum_ops import PolarizationSimulator
from utils.quantum_utils import QuantumStateHelper

class TestPolarizationSimulator(unittest.TestCase):
    
    def setUp(self):
        self.simulator = PolarizationSimulator()
    
    def test_circuit_creation(self):
        """Test quantum circuit creation"""
        qc = self.simulator.create_polarization_circuit(45, 0)
        
        # Verify circuit has correct number of qubits and gates
        self.assertEqual(qc.num_qubits, 1)
        self.assertEqual(qc.num_clbits, 1)
        
        # Verify circuit operations
        self.assertGreater(len(qc.data), 0)  # Should have at least measurement
    
    def test_theoretical_calculations(self):
        """Test theoretical probability calculations"""
        # Test |0⟩ state in rectilinear basis
        result = self.simulator.calculate_theoretical_probabilities(0, 0, 'rectilinear')
        self.assertAlmostEqual(result['prob_0'], 1.0, places=3)
        self.assertAlmostEqual(result['prob_1'], 0.0, places=3)
        
        # Test |+⟩ state in rectilinear basis
        result = self.simulator.calculate_theoretical_probabilities(90, 0, 'rectilinear')
        self.assertAlmostEqual(result['prob_0'], 0.5, places=3)
        self.assertAlmostEqual(result['prob_1'], 0.5, places=3)

class TestQuantumUtils(unittest.TestCase):
    
    def test_bloch_coordinates(self):
        """Test Bloch sphere coordinate conversion"""
        x, y, z = QuantumStateHelper.bloch_coordinates(0, 0)
        self.assertAlmostEqual(z, 1.0, places=3)  # |0⟩ state at north pole
        
        x, y, z = QuantumStateHelper.bloch_coordinates(180, 0)
        self.assertAlmostEqual(z, -1.0, places=3)  # |1⟩ state at south pole
    
    def test_binary_entropy(self):
        """Test binary entropy calculation"""
        # Test perfect cases
        self.assertEqual(QuantumStateHelper.binary_entropy(0), 0)
        self.assertEqual(QuantumStateHelper.binary_entropy(1), 0)
        
        # Test maximum entropy
        self.assertAlmostEqual(QuantumStateHelper.binary_entropy(0.5), 1.0, places=3)

if __name__ == '__main__':
    unittest.main()
