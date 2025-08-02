"""Global constants for the application"""

DEFAULT_SESSION_VALUES = {
    'learning_progress': {},
    'module_progress': {},
    'achievements': [],
    'experiment_history': [],
    'current_keys': {},
    'user_preferences': {
        'theme': 'light',
        'difficulty_level': 'beginner',
        'show_hints': True
    }
}

QUANTUM_CONSTANTS = {
    'PLANCK_CONSTANT': 6.62607015e-34,
    'LIGHT_SPEED': 299792458,
    'DEFAULT_WAVELENGTH': 1550e-9  # nm for telecom
}

MODULE_ORDER = [
    'polarization',
    'bb84', 
    'channel_errors',
    'eavesdropping',
    'error_correction',
    'security_analysis'
]
