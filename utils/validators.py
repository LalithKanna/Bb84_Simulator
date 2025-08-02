"""
Input validation utilities for quantum cryptography simulator
"""
from typing import Any, Dict, List, Union
import numpy as np


def validate_angle(angle: float) -> bool:
    """
    Validate angle parameter for quantum state preparation
    
    Args:
        angle: Angle value to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(angle, (int, float)) and 0 <= angle <= 180


def validate_shots(shots: int) -> bool:
    """
    Validate number of quantum measurement shots
    
    Args:
        shots: Number of shots to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(shots, int) and 1 <= shots <= 100000


def validate_phase(phi: float) -> bool:
    """
    Validate phase angle parameter
    
    Args:
        phi: Phase angle in degrees (0-360)
    
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(phi, (int, float)) and 0 <= phi <= 360


def validate_qber(qber: float) -> bool:
    """
    Validate Quantum Bit Error Rate
    
    Args:
        qber: Error rate value (0-0.5)
    
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(qber, (int, float)) and 0 <= qber <= 0.5


def validate_key_length(length: int) -> bool:
    """
    Validate key length for cryptographic protocols
    
    Args:
        length: Key length to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(length, int) and 4 <= length <= 1000


def validate_basis_choice(basis: str) -> bool:
    """
    Validate measurement basis selection
    
    Args:
        basis: Basis name to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    valid_bases = ['rectilinear', 'diagonal', 'circular']
    return isinstance(basis, str) and basis.lower() in valid_bases


def validate_probability(prob: float) -> bool:
    """
    Validate probability value
    
    Args:
        prob: Probability value to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(prob, (int, float)) and 0 <= prob <= 1


def validate_protocol_parameters(params: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate multiple protocol parameters at once
    
    Args:
        params: Dictionary of parameters to validate
    
    Returns:
        Dict[str, bool]: Validation results for each parameter
    """
    validation_results = {}
    
    for param, value in params.items():
        if param == 'theta':
            validation_results[param] = validate_angle(value)
        elif param == 'phi':
            validation_results[param] = validate_phase(value)
        elif param == 'shots':
            validation_results[param] = validate_shots(value)
        elif param == 'qber':
            validation_results[param] = validate_qber(value)
        elif param == 'key_length' or param == 'n_bits':
            validation_results[param] = validate_key_length(value)
        elif param == 'basis':
            validation_results[param] = validate_basis_choice(value)
        elif 'probability' in param.lower() or 'prob' in param.lower():
            validation_results[param] = validate_probability(value)
        else:
            # Default validation - just check if value exists
            validation_results[param] = value is not None
    
    return validation_results


def validate_quantum_state(state_vector: Union[List, np.ndarray]) -> bool:
    """
    Validate quantum state vector normalization
    
    Args:
        state_vector: Quantum state amplitudes
    
    Returns:
        bool: True if properly normalized, False otherwise
    """
    if not isinstance(state_vector, (list, np.ndarray)):
        return False
    
    try:
        state_array = np.array(state_vector, dtype=complex)
        norm = np.sum(np.abs(state_array) ** 2)
        return abs(norm - 1.0) < 1e-10  # Allow for numerical precision
    except:
        return False


def validate_circuit_parameters(n_qubits: int, n_clbits: int = None) -> bool:
    """
    Validate quantum circuit parameters
    
    Args:
        n_qubits: Number of qubits
        n_clbits: Number of classical bits (optional)
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(n_qubits, int) or n_qubits < 1 or n_qubits > 20:
        return False
    
    if n_clbits is not None:
        if not isinstance(n_clbits, int) or n_clbits < 0:
            return False
    
    return True


def validate_noise_level(noise: float) -> bool:
    """
    Validate noise level parameter
    
    Args:
        noise: Noise level (0-1)
    
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(noise, (int, float)) and 0 <= noise <= 1


def get_validation_error_message(param_name: str, param_value: Any) -> str:
    """
    Get descriptive error message for failed validation
    
    Args:
        param_name: Name of the parameter
        param_value: Value that failed validation
    
    Returns:
        str: Descriptive error message
    """
    error_messages = {
        'theta': f"Angle θ must be between 0° and 180°, got {param_value}",
        'phi': f"Phase φ must be between 0° and 360°, got {param_value}",
        'shots': f"Number of shots must be between 1 and 100,000, got {param_value}",
        'qber': f"QBER must be between 0 and 0.5, got {param_value}",
        'n_bits': f"Key length must be between 4 and 1000, got {param_value}",
        'basis': f"Basis must be 'rectilinear', 'diagonal', or 'circular', got {param_value}",
        'noise': f"Noise level must be between 0 and 1, got {param_value}"
    }
    
    return error_messages.get(param_name, f"Invalid value for {param_name}: {param_value}")


# Decorator for automatic parameter validation
def validate_params(**validation_rules):
    """
    Decorator to automatically validate function parameters
    
    Usage:
        @validate_params(theta=validate_angle, shots=validate_shots)
        def my_function(theta, shots):
            # Function code here
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function argument names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validation_rules.items():
                if param_name in bound_args.arguments:
                    param_value = bound_args.arguments[param_name]
                    if not validator(param_value):
                        raise ValueError(get_validation_error_message(param_name, param_value))
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Export commonly used validators
__all__ = [
    'validate_angle',
    'validate_shots', 
    'validate_phase',
    'validate_qber',
    'validate_key_length',
    'validate_basis_choice',
    'validate_probability',
    'validate_protocol_parameters',
    'validate_quantum_state',
    'validate_circuit_parameters',
    'validate_noise_level',
    'get_validation_error_message',
    'validate_params'
]
