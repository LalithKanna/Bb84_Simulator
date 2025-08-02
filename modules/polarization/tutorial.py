"""
Tutorial content for polarization module
"""
import streamlit as st
from typing import Dict, Any

# Add numpy import at the top
import numpy as np

class PolarizationTutorial:
    """Tutorial and educational content for polarization"""
    
    def render(self):
        """Render tutorial interface"""
        st.header("📚 Polarization Tutorial")
        
        st.markdown("""
        ### Welcome to Quantum Polarization!
        
        This tutorial will teach you about:
        - **Quantum state representation** on the Bloch sphere
        - **Bloch sphere visualization** and coordinates
        - **Measurement bases** (Rectilinear, Diagonal, Circular)
        - **Polarization experiments** and probability calculations
        
        ### Key Concepts:
        
        **🌐 Bloch Sphere**
        - Any qubit state can be represented as a point on a sphere
        - θ (theta): Polar angle from |0⟩ to |1⟩
        - φ (phi): Azimuthal angle for phase relationships
        
        **📏 Measurement Bases**
        - **Rectilinear**: Measures in {|0⟩, |1⟩} basis
        - **Diagonal**: Measures in {|+⟩, |-⟩} basis  
        - **Circular**: Measures in {|L⟩, |R⟩} basis
        
        **🎯 Learning Objectives**
        By the end of this module, you'll understand:
        - How quantum states are represented geometrically
        - The relationship between angles and measurement probabilities
        - How different measurement bases affect outcomes
        """)
        
        # Interactive tutorial section
        with st.expander("🎮 Quick Interactive Demo"):
            demo_theta = st.slider("Try θ angle:", 0, 180, 45, key="tutorial_theta")
            prob_0 = (np.cos(np.pi * demo_theta / 360)) ** 2
            prob_1 = 1 - prob_0
            
            st.write(f"For θ = {demo_theta}°:")
            st.write(f"- P(|0⟩) = {prob_0:.3f}")
            st.write(f"- P(|1⟩) = {prob_1:.3f}")
        
        if st.button("✅ Complete Tutorial & Start Experiments"):
            # Mark tutorial as completed in session state
            if 'module_progress' not in st.session_state:
                st.session_state.module_progress = {}
            if 'polarization' not in st.session_state.module_progress:
                st.session_state.module_progress['polarization'] = {}
            
            st.session_state.module_progress['polarization']['tutorial_completed'] = True
            st.success("🎉 Tutorial completed! Proceeding to main interface...")
            st.rerun()
    
    def render_contextual_hints(self, params: Dict[str, Any]):
        """Render contextual hints based on current parameters"""
        st.subheader("💡 Context Hints")
        
        theta = params.get('theta', 0)
        phi = params.get('phi', 0)
        basis = params.get('basis', 'Rectilinear')
        
        # Angle-specific hints
        if theta == 0:
            st.info("🔵 **|0⟩ State**: Pure ground state. Try increasing θ to add |1⟩ component!")
        elif theta == 90:
            st.info("🟢 **Superposition State**: Equal probability of |0⟩ and |1⟩!")
        elif theta == 180:
            st.info("🔴 **|1⟩ State**: Pure excited state.")
        elif 0 < theta < 90:
            st.info(f"🟡 **Mixed State**: More likely to measure |0⟩ ({((np.cos(np.pi*theta/360))**2):.2f} probability)")
        elif 90 < theta < 180:
            st.info(f"🟠 **Mixed State**: More likely to measure |1⟩ ({((np.sin(np.pi*theta/360))**2):.2f} probability)")
        
        # Phase hints
        if phi != 0:
            st.info(f"🌀 **Phase**: φ = {phi}° adds quantum phase (affects interference)")
        
        # Basis hints
        if basis == "Diagonal":
            st.info("📐 **Diagonal Basis**: Measuring in {|+⟩, |-⟩} - different from computational basis")
        elif basis == "Circular":
            st.info("🌀 **Circular Basis**: Measuring circular polarizations {|L⟩, |R⟩}")
