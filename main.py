"""
Main application entry point for Quantum Cryptography Simulator
"""
import streamlit as st
from core.session_manager import SessionManager
from core.achievement_system import AchievementSystem
from components.layouts import MainLayout
from config.settings import APP_CONFIG
from modules import (
    polarization,
    bb84, 
    channel_errors,
    eavesdropping,
    error_correction,
    security_analysis
)

def main():
    """Main application function"""
    
    # Initialize Streamlit configuration
    st.set_page_config(**APP_CONFIG['streamlit'])
    
    # Initialize session management
    session_manager = SessionManager()
    session_manager.initialize()
    
    # Initialize achievement system
    achievement_system = AchievementSystem()
    
    # Create main layout
    layout = MainLayout()
    layout.render_header()
    
    # Module routing
    module_map = {
        "Quantum States & Polarization": polarization.main,
        "BB84 Protocol Basics": bb84.main,
        "Channel Errors & Noise": channel_errors.main,
        "Eavesdropping Detection": eavesdropping.main,
        "Error Correction": error_correction.main,
        "Security Analysis": security_analysis.main
    }
    
    # Render sidebar and get selected module
    selected_module = layout.render_sidebar(list(module_map.keys()))
    
    # Execute selected module
    try:
        module_function = module_map[selected_module]
        module_function()
        
        # Update achievements
        achievement_system.check_module_completion(selected_module)
        
    except Exception as e:
        st.error(f"Module error: {str(e)}")
        st.info("Please refresh or contact support if the issue persists.")
    
    # Render gamification sidebar
    achievement_system.render_sidebar()
    
    # Render footer
    layout.render_footer()

if __name__ == "__main__":
    main()
