"""
Centralized session state management for all modules
"""
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List
from config.constants import DEFAULT_SESSION_VALUES

class SessionManager:
    """Manages application-wide session state"""
    
    def __init__(self):
        self.session_keys = [
            'learning_progress',
            'module_progress', 
            'achievements',
            'experiment_history',
            'current_keys',
            'user_preferences'
        ]
    
    def initialize(self) -> None:
        """Initialize all session state variables"""
        for key in self.session_keys:
            if key not in st.session_state:
                st.session_state[key] = DEFAULT_SESSION_VALUES.get(key, {})
    
    def get_module_progress(self, module_name: str) -> Dict[str, Any]:
        """Get progress for specific module"""
        return st.session_state.module_progress.get(module_name, {})
    
    def update_module_progress(self, module_name: str, progress_data: Dict[str, Any]) -> None:
        """Update progress for specific module"""
        if 'module_progress' not in st.session_state:
            st.session_state.module_progress = {}
        
        st.session_state.module_progress[module_name] = progress_data
    
    def add_experiment_record(self, module_name: str, experiment_data: Dict[str, Any]) -> None:
        """Add experiment to history"""
        experiment_record = {
            'module': module_name,
            'timestamp': datetime.now().isoformat(),
            **experiment_data
        }
        
        if 'experiment_history' not in st.session_state:
            st.session_state.experiment_history = []
            
        st.session_state.experiment_history.append(experiment_record)
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference with fallback"""
        return st.session_state.get('user_preferences', {}).get(key, default)
    
    def set_user_preference(self, key: str, value: Any) -> None:
        """Set user preference"""
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
        
        st.session_state.user_preferences[key] = value
    
    def reset_module_progress(self, module_name: str) -> None:
        """Reset progress for specific module"""
        if module_name in st.session_state.get('module_progress', {}):
            del st.session_state.module_progress[module_name]
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export session data for backup/analysis"""
        return {
            key: st.session_state.get(key, {}) 
            for key in self.session_keys
            if key in st.session_state
        }
