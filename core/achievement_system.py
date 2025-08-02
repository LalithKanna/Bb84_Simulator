"""Simple achievement system for educational gamification"""
import streamlit as st
from typing import List, Dict, Any

class AchievementSystem:
    def __init__(self):
        self.achievements = {
            'first_experiment': {'name': '🔬 First Experiment', 'description': 'Run your first quantum experiment'},
            'polarization_master': {'name': '🌐 Polarization Master', 'description': 'Complete all polarization exercises'},
            'bb84_novice': {'name': '🔐 BB84 Novice', 'description': 'Successfully run BB84 protocol'},
        }
    
    def check_module_completion(self, module_name: str):
        """Check and award achievements for module completion"""
        if 'achievements' not in st.session_state:
            st.session_state.achievements = []
        
        # Simple achievement logic
        if module_name == 'polarization' and 'polarization_master' not in st.session_state.achievements:
            st.session_state.achievements.append('polarization_master')
            st.success("🎉 Achievement Unlocked: Polarization Master!")
    
    def render_sidebar(self):
        """Render achievements in sidebar"""
        with st.sidebar:
            st.subheader("🏆 Achievements")
            earned = st.session_state.get('achievements', [])
            if earned:
                for achievement_id in earned:
                    achievement = self.achievements.get(achievement_id, {})
                    st.write(f"{achievement.get('name', 'Unknown')} ✅")
            else:
                st.write("No achievements yet!")
