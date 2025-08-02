"""
Quantum operations for polarization experiments
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from typing import Dict, Any, Tuple
from config.settings import APP_CONFIG

class PolarizationSimulator:
    """Handles quantum simulations for polarization experiments"""
    
    def __init__(self):
        self.backend = AerSimulator()
        self.default_shots = APP_CONFIG['quantum']['default_shots']
    
    def create_polarization_circuit(self, theta: float, phi: float) -> QuantumCircuit:
        """Create quantum circuit for polarization state"""
        
        qc = QuantumCircuit(1, 1)
        
        # State preparation
        if theta != 0:
            qc.ry(2 * np.pi * theta / 180, 0)
        
        if phi != 0:
            qc.rz(phi, 0)
        
        qc.measure(0, 0)
        
        return qc
    
    def run_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete polarization experiment"""
        
        # Extract parameters
        theta = params.get('theta', 0)
        phi = params.get('phi', 0)
        basis = params.get('basis', 'Rectilinear')
        shots = params.get('shots', self.default_shots)
        
        # Create circuit
        qc = self.create_polarization_circuit(theta, phi)
        
        # Add basis rotation for measurement
        if basis.lower() == 'diagonal':
            qc.h(0)
        elif basis.lower() == 'circular':
            qc.sdg(0)
            qc.h(0)
        
        # Execute simulation
        job = self.backend.run(transpile(qc, self.backend), shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Process results
        total_shots = sum(counts.values())
        prob_0 = counts.get('0', 0) / total_shots
        prob_1 = counts.get('1', 0) / total_shots
        
        # Calculate theoretical values
        theoretical = self.calculate_theoretical_probabilities(theta, phi, basis)
        
        return {
            'experimental': {
                'counts': counts,
                'probabilities': {'0': prob_0, '1': prob_1},
                'total_shots': total_shots
            },
            'theoretical': theoretical,
            'parameters': params,
            'circuit': qc,
            'accuracy': 1 - abs(prob_0 - theoretical['prob_0'])
        }
    
    def calculate_theoretical_probabilities(self, theta: float, phi: float, basis: str) -> Dict[str, float]:
        """Calculate theoretical measurement probabilities"""
        
        theta_rad = np.pi * theta / 180
        
        if basis.lower() == 'rectilinear':
            prob_0 = np.cos(theta_rad / 2) ** 2
        elif basis.lower() == 'diagonal':
            prob_0 = 0.5 * (1 + np.sin(theta_rad))
        else:  # circular
            prob_0 = 0.5 * (1 + np.cos(theta_rad))
        
        return {
            'prob_0': prob_0,
            'prob_1': 1 - prob_0
        }
