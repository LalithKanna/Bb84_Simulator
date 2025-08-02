"""
Reusable UI components for consistent interface design
"""
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from config.settings import APP_CONFIG

class ParameterControls:
    """Reusable parameter control widgets"""
    
    def render_polarization_controls(self) -> Dict[str, Any]:
        """Render polarization parameter controls"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            theta = st.slider(
                "Polarization Angle (θ)", 
                min_value=0, max_value=180,
                value=st.session_state.get('theta', 45),
                step=5,
                help="Angle from |0⟩ state on Bloch sphere"
            )
        
        with col2:
            phi = st.slider(
                "Azimuthal Angle (φ)",
                min_value=0, max_value=360,
                value=st.session_state.get('phi', 0),
                step=5,
                help="Phase angle on Bloch sphere"
            )
        
        basis = st.selectbox(
            "Measurement Basis",
            ["Rectilinear", "Diagonal", "Circular"],
            index=0,
            help="Basis for quantum measurement"
        )
        
        shots = st.select_slider(
            "Number of Measurements",
            options=[100, 500, 1000, 2000, 5000],
            value=1000
        )
        
        return {
            'theta': theta,
            'phi': phi, 
            'basis': basis,
            'shots': shots
        }
    
    def render_bb84_controls(self) -> Dict[str, Any]:
        """Render BB84 protocol controls"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_bits = st.slider(
                "Number of Qubits",
                min_value=4, max_value=50,
                value=12,
                help="Number of qubits to send in protocol"
            )
        
        with col2:
            noise_level = st.slider(
                "Channel Noise (%)",
                min_value=0, max_value=25,
                value=0,
                help="Percentage of noise in quantum channel"
            ) / 100
        
        return {
            'n_bits': n_bits,
            'noise_level': noise_level
        }

class ResultsDisplay:
    """Standardized results display components"""
    
    def show_polarization_results(self, results: Dict[str, Any]) -> None:
        """Display polarization experiment results"""
        
        st.subheader("📊 Experiment Results")
        
        exp_results = results['experimental']
        theo_results = results['theoretical']
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Measured |0⟩",
                f"{exp_results['probabilities']['0']:.2%}",
                delta=f"Theory: {theo_results['prob_0']:.2%}"
            )
        
        with col2:
            st.metric(
                "Measured |1⟩", 
                f"{exp_results['probabilities']['1']:.2%}",
                delta=f"Theory: {theo_results['prob_1']:.2%}"
            )
        
        with col3:
            st.metric(
                "Accuracy",
                f"{results['accuracy']:.1%}",
                delta="vs Theory"
            )
        
        # Counts table
        counts_df = pd.DataFrame([
            {"Outcome": "|0⟩", "Count": exp_results['counts'].get('0', 0)},
            {"Outcome": "|1⟩", "Count": exp_results['counts'].get('1', 0)}
        ])
        
        st.dataframe(counts_df, use_container_width=True)
    
    def show_bb84_results(self, results: Dict[str, Any]) -> None:
        """Display BB84 protocol results"""
        
        st.subheader("🔐 BB84 Protocol Results")
        
        # Implementation for BB84 specific results
        pass

class ModuleHeader:
    """Standardized module header component"""
    
    @staticmethod
    def render(title: str, description: str, icon: str, gradient_colors: List[str] = None) -> None:
        """Render module header with consistent styling"""
        
        if gradient_colors is None:
            gradient_colors = ['#4a90e2', '#357abd']
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {gradient_colors[0]} 0%, {gradient_colors[1]} 100%); 
            padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        ">
            <h1 style="margin: 0; font-size: 2.2em; font-weight: 700;">
                {icon} {title}
            </h1>
            <p style="margin: 1rem 0 0 0; font-size: 1.1em; opacity: 0.95;">
                {description}
            </p>
        </div>
        """, unsafe_allow_html=True)
