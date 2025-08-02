"""Layout components for consistent UI design"""
import streamlit as st
from typing import List

class MainLayout:
    """Main application layout manager"""
    
    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.5em;">🔬 Quantum Cryptography Simulator</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2em;">Interactive Education Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self, modules: List[str]) -> str:
        """Render sidebar navigation"""
        with st.sidebar:
            st.title("📚 Learning Modules")
            selected = st.radio("Choose Module:", modules)
            
            st.divider()
            st.subheader("📊 Quick Stats")
            total_experiments = len(st.session_state.get('experiment_history', []))
            st.metric("Total Experiments", total_experiments)
            
            return selected
    
    def render_footer(self):
        """Render application footer"""
        st.divider()
        st.markdown("""
        <div style="text-align: center; padding: 1rem; color: #666;">
            Built with ❤️ using Streamlit, Qiskit & QuTiP | Educational Use Only
        </div>
        """, unsafe_allow_html=True)
