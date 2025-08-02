"""
Application configuration settings
"""

APP_CONFIG = {
    'streamlit': {
        'page_title': "Interactive QKD Simulator",
        'page_icon': "🔬",
        'layout': "wide",
        'initial_sidebar_state': "expanded"
    },
    'quantum': {
        'default_shots': 1000,
        'max_qubits': 10,
        'simulator_backend': 'aer_simulator'
    },
    'ui': {
        'animation_speed': 0.1,
        'color_scheme': {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545'
        }
    },
    'security': {
        'max_qber': 0.11,
        'default_security_parameter': 1e-8,
        'key_rate_threshold': 0.1
    },
    'achievements': {
        'enable_notifications': True,
        'save_progress': True,
        'leaderboard_enabled': False
    }
}

MODULES_CONFIG = {
    'polarization': {
        'max_angle': 180,
        'default_shots': 1000,
        'animation_enabled': True
    },
    'bb84': {
        'min_bits': 4,
        'max_bits': 50,
        'default_bits': 12
    },
    'eavesdropping': {
        'attack_types': ['intercept_resend', 'beam_splitting', 'photon_number_splitting'],
        'detection_threshold': 0.05
    }
}
