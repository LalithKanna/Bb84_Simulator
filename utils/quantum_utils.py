"""
Shared quantum computing utilities across all modules
"""
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from typing import List, Dict, Any, Tuple
from config.constants import QUANTUM_CONSTANTS

class QuantumStateHelper:
    """Helper class for quantum state operations"""
    
    @staticmethod
    def bloch_coordinates(theta: float, phi: float) -> Tuple[float, float, float]:
        """Convert spherical to Cartesian coordinates for Bloch sphere"""
        theta_rad = np.pi * theta / 180
        phi_rad = np.pi * phi / 180
        
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)
        
        return x, y, z
    
    @staticmethod
    def binary_entropy(x: float) -> float:
        """Calculate binary entropy function"""
        if x <= 0 or x >= 1:
            return 0
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    
    @staticmethod
    def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculate fidelity between two quantum states"""
        return abs(np.vdot(state1, state2)) ** 2

class CircuitBuilder:
    """Advanced quantum circuit building utilities"""
    
    @staticmethod
    def create_bell_state(state_type: str = 'phi_plus') -> QuantumCircuit:
        """Create Bell state circuits"""
        
        qc = QuantumCircuit(2, 2)
        
        # Create |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        qc.h(0)
        qc.cx(0, 1)
        
        # Modify for other Bell states
        if state_type == 'phi_minus':
            qc.z(0)
        elif state_type == 'psi_plus':
            qc.x(1)
        elif state_type == 'psi_minus':
            qc.x(1)
            qc.z(0)
        
        qc.measure_all()
        return qc
    
    @staticmethod
    def add_noise_model(qc: QuantumCircuit, noise_params: Dict[str, float]) -> QuantumCircuit:
        """Add noise model to quantum circuit"""
        # Implementation for adding various noise types
        # This is a placeholder for full noise model implementation
        return qc

def validate_quantum_params(**kwargs) -> Dict[str, bool]:
    """Validate quantum parameters"""
    validation_results = {}
    
    for param, value in kwargs.items():
        if param == 'theta':
            validation_results[param] = 0 <= value <= 180
        elif param == 'phi':
            validation_results[param] = 0 <= value <= 360
        elif param == 'shots':
            validation_results[param] = 1 <= value <= 100000
        elif param == 'qber':
            validation_results[param] = 0 <= value <= 0.5
        else:
            validation_results[param] = True
    
    return validation_results
