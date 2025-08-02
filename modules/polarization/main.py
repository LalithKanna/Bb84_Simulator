import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, Any, List, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import json
from datetime import datetime
import scipy.stats as stats
from .quantum_ops import PolarizationSimulator
from .tutorial import PolarizationTutorial
from core.session_manager import SessionManager
from utils.validators import validate_angle, validate_shots
from components.ui_components import ParameterControls, ResultsDisplay
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, List

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ────────────────────────────────────────────────────────────────────────────────
#  NEW THEORY & DYNAMIC-EXPLANATION CLASSES
# ────────────────────────────────────────────────────────────────────────────────
class TheoryContent:
    """Renders progressively layered theory panels."""

    # --------------------------------------------------------------------------
    # High-level concept introductions
    # --------------------------------------------------------------------------
    def render_concept_introduction(self, concept: str, level: str) -> None:
        if concept == "polarization_basics":
            self._intro_polarization(level)
        elif concept == "measurement":
            self._intro_measurement(level)
        elif concept == "bloch":
            self._intro_bloch(level)
        elif concept == "qkd":
            self._intro_qkd(level)

    # --------------------------------------------------------------------------
    # Layer-specific mathematical foundations
    # --------------------------------------------------------------------------
    def render_mathematical_foundation(self, level: str) -> None:
        if level == "beginner":
            st.latex(r"|\,\psi\rangle = \alpha\,|0\rangle + \beta\,|1\rangle")
            st.markdown(
                "Here **α** and **β** are complex numbers whose squared magnitudes give "
                "the probabilities of measuring the photon in the vertical (|0⟩) or "
                "horizontal (|1⟩) state."
            )
        elif level == "intermediate":
            st.markdown(
                "_Born’s rule_:  \n"
                r"\(P(|0\rangle)=|\alpha|^2,\;\;P(|1\rangle)=|\beta|^2\)", unsafe_allow_html=True
            )
            st.latex(r"\alpha = \cos\frac{\theta}{2},\qquad \beta = e^{i\phi}\sin\frac{\theta}{2}")
            st.markdown(
                "These relations map directly onto the Bloch‐sphere coordinates "
                "with polar angle θ and azimuthal angle φ."
            )
        else:  # advanced
            st.markdown("**Unitary rotations** on the Bloch sphere:")
            st.latex(r"U(\hat{n},\theta)=e^{-i\theta\hat{n}\cdot\vec{\sigma}/2}")
            st.markdown(
                "where \(\\vec{\\sigma}\) are Pauli matrices and **ẑ**, **x̂**, **ŷ** "
                "correspond to rectilinear, diagonal, and circular bases."
            )

    # --------------------------------------------------------------------------
    # Real-world application contexts
    # --------------------------------------------------------------------------
    def render_applications_context(self, concept: str) -> None:
        if concept == "polarization_basics":
            st.markdown(
                "• **Quantum communications** use single-photon polarization to carry qubit "
                "information through fibres or free-space links.  \n"
                "• In **microscopy** and **remote sensing**, polarization contrast enhances "
                "material discrimination beyond intensity-only images."
            )
        elif concept == "qkd":
            st.markdown(
                "Polarization underpins **BB84 QKD**. Orthogonal bases prevent an eavesdropper "
                "from measuring without introducing errors that Alice and Bob can detect."
            )

    # --------------------------------------------------------------------------
    # Deeper principles on demand
    # --------------------------------------------------------------------------
    def render_quantum_principles(self, principle: str) -> None:
        if principle == "uncertainty":
            st.latex(r"\sigma_X\sigma_Z \ge \frac{1}{2}|\langle[Y,Z]\rangle|")
            st.markdown(
                "Measuring in incompatible bases trades precision: knowing polarization "
                "perfectly in one basis maximises uncertainty in the conjugate basis."
            )

    # --------------------------------------------------------------------------
    # Internal helper blocks
    # --------------------------------------------------------------------------
    def _intro_polarization(self, level: str) -> None:
        st.subheader("🔍 What is Photon Polarization?")
        st.markdown(
            "**Polarization** describes the oscillation direction of the electric-field "
            "vector. In quantum mechanics each single photon behaves as a two-state system, "
            "making polarization a natural qubit."
        )
        self.render_mathematical_foundation(level)
        self.render_applications_context("polarization_basics")

    def _intro_measurement(self, level: str) -> None:
        st.subheader("📐 Measurement Theory Refresher")
        st.markdown(
            "Selecting a measurement basis corresponds to rotating the Bloch sphere "
            "before the **projective measurement**. The state _collapses_ onto |0⟩ or |1⟩."
        )
        if level != "beginner":
            self.render_quantum_principles("uncertainty")

    def _intro_bloch(self, level: str) -> None:
        st.subheader("🌀 Visualising States on the Bloch Sphere")
        st.markdown(
            "Any qubit state maps to a point on the unit sphere. Rotations reflect "
            "unitary operations, and latitude encodes population while longitude "
            "encodes relative phase."
        )
        self.render_mathematical_foundation(level)

    def _intro_qkd(self, level: str) -> None:
        st.subheader("🔒 Quantum Key Distribution (BB84)")
        st.markdown(
            "Four polarization states in two mutually unbiased bases carry raw key bits. "
            "Measurement errors above a threshold signal eavesdropping."
        )
        if level == "advanced":
            st.latex(r"P_{\text{err}}^{\text{Eve}}\ge \frac{1}{4}")
# ────────────────────────────────────────────────────────────────────────────────


class DynamicExplanations:
    """Generates context-aware, real-time explanations."""

    def __init__(self):
        self._last_params: Dict[str, float] = {}

    # --------------------------------------------------------------------------
    # Before interaction
    # --------------------------------------------------------------------------
    def before_interaction(self, concept: str) -> None:
        st.info(f"• _Ready to explore **{concept.replace('_', ' ').title()}**? "
                "Adjust the controls to predict outcomes before running the experiment._")

    # --------------------------------------------------------------------------
    # During interaction (sliders, etc.)
    # --------------------------------------------------------------------------
    def explain_parameter_change(self, parameter: str, new: float) -> None:
        old = self._last_params.get(parameter, new)
        if new != old:
            delta = new - old
            direction = "increased" if delta > 0 else "decreased"
            st.write(f"🔄 **{parameter.upper()}** {direction} to {new:.1f}. "
                     f"This will rotate the Bloch vector {abs(delta):.1f}°.")
            self._last_params[parameter] = new

    # --------------------------------------------------------------------------
    # After experiment completes
    # --------------------------------------------------------------------------
    def interpret_results(self,
                          counts: Dict[str, int],
                          theoretical: Dict[str, float],
                          basis: str) -> None:
        total = sum(counts.values())
        exp_p0 = counts.get("0", 0) / total
        th_p0 = theoretical["0"]
        error = abs(exp_p0 - th_p0)
        st.success(
            f"✅ Experimental probability P(|0⟩) = {exp_p0:.2f} "
            f"(theory {th_p0:.2f}). Deviation = {error:.2%}."
        )
        if error > 0.1:
            st.warning(
                "Noticeable deviation—try increasing **shots** or checking for "
                "noise sources."
            )

    # --------------------------------------------------------------------------
    # Learning pathway suggestions
    # --------------------------------------------------------------------------
    def provide_next_steps(self, level: str) -> None:
        if level == "beginner":
            st.info("Next ➡️ Vary **φ** to see how phase only affects diagonal/circular bases.")
        elif level == "intermediate":
            st.info("Next ➡️ Compare outcome statistics across two bases to visualise uncertainty.")
        else:
            st.info("Next ➡️ Enable noise modelling and investigate security implications for BB84.")

class EnhancedPolarizationModule:
    """Enhanced polarization module with advanced learning analytics"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.simulator = PolarizationSimulator()
        self.tutorial = PolarizationTutorial()
        self.backend = AerSimulator()
        self.theory = TheoryContent()
        self.dynamic = DynamicExplanations()
        if 'polarization_performance' not in st.session_state:
            st.session_state.polarization_performance = []
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {'level': 'beginner'}
    def main(self) -> None:
        st.set_page_config(page_title="Quantum Polarization Lab", layout="wide")
        self._hero_banner()
        
        try:
            user_perf = st.session_state.get("polarization_performance", [])
            learning = self.analyze_learning_pattern(user_perf)
            
            # Ensure all required keys exist to prevent KeyError
            learning.setdefault('level', 'beginner')
            learning.setdefault('focus_areas', [])
            learning.setdefault('recent_accuracy', 0.0)
            learning.setdefault('total_attempts', len(user_perf))
            
        except Exception as e:
            st.error(f"Error in learning analysis: {str(e)}")
            # Fallback learning analysis
            learning = {
                'level': 'beginner',
                'focus_areas': [],
                'recent_accuracy': 0.0,
                'total_attempts': 0
            }
            user_perf = []

        # Side bar info
        st.sidebar.header("👤 Personalised Tracking")
        st.sidebar.write(f"**Level:** {learning['level'].title()}")
        st.sidebar.write(f"Attempts: {learning['total_attempts']}")
        if learning.get("focus_areas"):
            st.sidebar.write("Focus areas:")
            for a in learning["focus_areas"]:
                st.sidebar.write(f"• {a}")

        # Tabs
        tabs = st.tabs([
            "📚 Theory-First Lab",
            "🌐 Bloch Sphere", 
            "📊 Basis Comparison",
            "🧪 Quantum Circuits",
            "📈 Learning Analytics",
            "🎬 Animations"
        ])

        with tabs[0]:
            try:
                self._panel_theory_intro("polarization_basics", learning["level"])
                self.render_interactive_lab(learning)
            except Exception as e:
                st.error(f"Error in Theory-First Lab: {str(e)}")
                st.info("Please refresh the page or contact support if the issue persists.")

        with tabs[1]:
            try:
                self._panel_theory_intro("bloch", learning["level"])
                self.render_3d_bloch_sphere()
            except Exception as e:
                st.error(f"Error in Bloch Sphere visualization: {str(e)}")

        with tabs[2]:
            try:
                self._panel_theory_intro("measurement", learning["level"])
                self.render_comparative_analysis()
            except Exception as e:
                st.error(f"Error in Basis Comparison: {str(e)}")

        with tabs[3]:
            try:
                self._panel_theory_intro("qkd", learning["level"])
                self.render_quantum_circuits()
            except Exception as e:
                st.error(f"Error in Quantum Circuits: {str(e)}")

        with tabs[4]:
            try:
                self.render_learning_analytics(user_perf, learning)
            except Exception as e:
                st.error(f"Error in Learning Analytics: {str(e)}")

        with tabs[5]:
            try:
                self.render_animations()
            except Exception as e:
                st.error(f"Error in Animations: {str(e)}")

    def _panel_theory_intro(self, concept: str, level: str) -> None:
        with st.expander("📖 Theory Overview", expanded=True):
            self.theory.render_concept_introduction(concept, level)  
    def render_interactive_lab(self, learning_analysis: Dict):
        """Enhanced interactive laboratory with personalized content"""
        st.header("ðŸŽ¯ Personalized Interactive Laboratory")
        
        # Get personalized content based on learning level
        personalized_content = self.get_personalized_content(learning_analysis)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("âš™ï¸ Quantum State Parameters")
            
            # Adaptive parameter ranges based on learning level
            if learning_analysis['level'] == 'beginner':
                theta_step = 15
                theta_help = "Start with common angles: 0Â°, 45Â°, 90Â°"
            elif learning_analysis['level'] == 'intermediate':
                theta_step = 5
                theta_help = "Explore intermediate angles for deeper understanding"
            else:
                theta_step = 1
                theta_help = "Fine-tune angles for advanced analysis"
            
            theta = st.slider(
                "**Polar Angle Î¸** (degrees)", 
                min_value=0, max_value=180, value=45, step=theta_step,
                help=theta_help
            )
            
            phi = st.slider(
                "**Azimuthal Angle Ï†** (degrees)", 
                min_value=0, max_value=360, value=0, step=theta_step,
                help="Controls quantum phase relationships"
            )
            
            basis = st.selectbox(
                "**Measurement Basis**",
                options=["Rectilinear", "Diagonal", "Circular"],
                help="Choose measurement basis based on your learning level"
            )
            
            shots = st.select_slider(
                "**Number of Measurements**",
                options=[100, 500, 1000, 5000, 10000],
                value=1000
            )
            
            # Personalized experiment suggestions
            st.subheader("ðŸ’¡ Recommended Experiments")
            for i, experiment in enumerate(personalized_content['experiments']):
                if st.button(f"ðŸ§ª {experiment}", key=f"exp_{i}"):
                    st.info(f"Try this experiment: {experiment}")
            
            if st.button("ðŸš€ Run Enhanced Simulation", type="primary"):
                self.run_enhanced_simulation(theta, phi, basis, shots, learning_analysis)
        
        with col2:
            # Quick presets with learning level adaptation
            st.subheader("âš¡ Smart Presets")
            
            if learning_analysis['level'] == 'beginner':
                presets = [
                    ("ðŸ”´ |0âŸ© Ground State", 0, 0, "Rectilinear"),
                    ("ðŸ”µ |1âŸ© Excited State", 180, 0, "Rectilinear"),
                    ("ðŸŸ¢ |+âŸ© Plus State", 90, 0, "Diagonal"),
                    ("ðŸŸ¡ |-âŸ© Minus State", 90, 180, "Diagonal")
                ]
            else:
                presets = [
                    ("ðŸ”´ |0âŸ©", 0, 0, "Rectilinear"),
                    ("ðŸ”µ |1âŸ©", 180, 0, "Rectilinear"),
                    ("ðŸŸ¢ |+âŸ©", 90, 0, "Diagonal"),
                    ("ðŸŸ¡ |-âŸ©", 90, 180, "Diagonal"),
                    ("ðŸŸ  |+iâŸ©", 90, 90, "Circular"),
                    ("ðŸŸ£ |-iâŸ©", 90, 270, "Circular")
                ]
            
            for i, (label, t, p, b) in enumerate(presets):
                if st.button(label, key=f"preset_{i}"):
                    st.session_state['theta'] = t
                    st.session_state['phi'] = p
                    st.session_state['basis'] = b
                    st.rerun()
            
            # Learning concepts for current level
            st.subheader("ðŸ§  Key Concepts")
            for concept in personalized_content['concepts']:
                st.write(f"â€¢ {concept}")
    
    def render_3d_bloch_sphere(self):
        """Interactive 3D Bloch sphere visualization"""
        st.header("ðŸŒ Interactive 3D Bloch Sphere")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("ðŸŽ›ï¸ Controls")
            theta = st.slider("Î¸ (Polar)", 0, 180, 45, 1)
            phi = st.slider("Ï† (Azimuthal)", 0, 360, 0, 5)
            
            show_projections = st.checkbox("Show Projections", value=True)
            show_trajectories = st.checkbox("Show Trajectories", value=False)
        
        with col1:
            # Create enhanced 3D Bloch sphere
            fig = self.create_interactive_bloch_sphere(theta, phi, show_projections)
            st.plotly_chart(fig, use_container_width=True)
            
            # State information panel
            self.display_state_information(theta, phi)
    
    def render_comparative_analysis(self):
        """Comparative analysis between different bases"""
        st.header("ðŸ“Š Comparative Basis Analysis")
        
        theta = st.slider("**Angle for Comparison**", 0, 180, 45, 5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            basis1 = st.selectbox("**First Basis**", ["Rectilinear", "Diagonal", "Circular"])
        
        with col2:
            basis2 = st.selectbox("**Second Basis**", ["Diagonal", "Rectilinear", "Circular"])
        
        if st.button("ðŸ“ˆ Generate Comparison"):
            fig = self.create_comparative_visualization(theta, basis1, basis2)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis
            self.display_comparative_analysis(theta, basis1, basis2)
    
    def render_quantum_circuits(self):
        """Quantum circuit visualization and simulation"""
        st.header("ðŸ§ª Quantum Circuits & Advanced Simulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("ðŸ”§ Circuit Parameters")
            theta = st.slider("State Angle", 0, 180, 45)
            basis = st.selectbox("Measurement Basis", ["Rectilinear", "Diagonal", "Circular"])
            
            # Noise modeling
            st.subheader("ðŸŒŠ Noise Modeling")
            add_noise = st.checkbox("Add Realistic Noise")
            
            noise_params = {}
            if add_noise:
                noise_params['depolarizing_error'] = st.slider("Depolarizing Error", 0.0, 0.1, 0.01)
                noise_params['measurement_error'] = st.slider("Measurement Error", 0.0, 0.05, 0.005)
            
            shots = st.slider("Circuit Shots", 100, 10000, 1000)
            
            if st.button("âš¡ Run Circuit Simulation"):
                self.run_circuit_simulation(theta, basis, noise_params, shots)
        
        with col2:
            # Display quantum circuit
            qc = self.create_enhanced_quantum_circuit(theta, basis, noise_params)
            
            # Circuit diagram (simplified representation)
            st.subheader("ðŸ“‹ Quantum Circuit")
            st.code(f"""
            QuantumCircuit(1, 1)
            â”œâ”€ RY({2*np.pi*theta/180:.3f}) â”€â”¤
            â”œâ”€ Measurement Basis: {basis}
            â””â”€ Measure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
            """)
    
    def render_learning_analytics(self, user_performance: List[Dict], analysis: Dict):
        """Advanced learning analytics dashboard"""
        st.header("ðŸ“ˆ AI-Powered Learning Analytics")
        
        if not user_performance:
            st.info("ðŸŽ¯ Complete some experiments to see your learning analytics!")
            return
        
        # Performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Learning Level", analysis['level'].title())
        
        with col2:
            st.metric("Recent Accuracy", f"{analysis['recent_accuracy']:.1%}")
        
        with col3:
            st.metric("Total Attempts", analysis['total_attempts'])
        
        with col4:
            improvement = self.calculate_improvement_trend(user_performance)
            st.metric("Improvement Trend", f"{improvement:+.1%}")
        
        # Detailed analytics
        st.subheader("ðŸ“Š Performance Trends")
        self.create_performance_dashboard(user_performance)
        
        # Personalized recommendations
        st.subheader("ðŸŽ¯ AI Recommendations")
        recommendations = self.generate_ai_recommendations(analysis, user_performance)
        for rec in recommendations:
            st.write(f"â€¢ {rec}")
    
    def render_animations(self):
        """Animated visualizations for state evolution"""
        st.header("ðŸŽ¬ Quantum State Animations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ðŸŽ­ Animation Controls")
            start_theta = st.slider("Start Angle", 0, 180, 0)
            end_theta = st.slider("End Angle", 0, 180, 180)
            animation_steps = st.slider("Animation Steps", 5, 50, 20)
            animation_speed = st.slider("Speed (seconds per frame)", 0.1, 2.0, 0.2)
            
            if st.button("ðŸŽ¬ Start Animation"):
                self.create_animated_bloch_sphere(start_theta, end_theta, animation_steps, animation_speed)
        
        with col2:
            st.subheader("ðŸ“š Animation Types")
            st.write("ðŸ”„ **State Evolution**: Watch quantum states transform")
            st.write("ðŸŒ€ **Basis Rotation**: See measurement basis changes")
            st.write("âš¡ **Quantum Gates**: Visualize gate operations")
    
    # =============================================================================
    # ENHANCED LEARNING ANALYTICS FUNCTIONS
    # =============================================================================
    
    @staticmethod
    def analyze_learning_pattern(user_performance: List[Dict]) -> Dict:
        """Analyze user's learning patterns and identify areas for improvement"""
        if not user_performance:
            return {
                'level': 'beginner', 
                'focus_areas': ['basic_concepts'],
                'recent_accuracy': 0.0,
                'total_attempts': 0
            }
        
        # Analyze recent performance
        recent_accuracy = np.mean([p['accuracy'] for p in user_performance[-5:]])
        concept_struggles = {}
        
        for perf in user_performance:
            if perf['accuracy'] < 0.7:
                concept = perf.get('concept', 'general')
                concept_struggles[concept] = concept_struggles.get(concept, 0) + 1
        
        # Determine difficulty level
        if recent_accuracy > 0.9:
            level = 'advanced'
        elif recent_accuracy > 0.7:
            level = 'intermediate'
        else:
            level = 'beginner'
        
        # Identify focus areas
        focus_areas = sorted(concept_struggles.keys(), key=concept_struggles.get, reverse=True)[:3]
        
        return {
            'level': level,
            'focus_areas': focus_areas,
            'recent_accuracy': recent_accuracy,
            'total_attempts': len(user_performance)  # This was missing in empty case
        }
    def _hero_banner(self) -> None:
        st.markdown("""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                padding:1.5rem;border-radius:12px;color:white;text-align:center;"
         role="banner" aria-label="Quantum Polarization Laboratory">
      <!-- existing content -->
    </div>
    """, unsafe_allow_html=True)

    
    @staticmethod
    def get_personalized_content(analysis: Dict) -> Dict:
        """Generate personalized learning content based on analysis"""
        content = {
            'beginner': {
                'concepts': ['Basic polarization', 'Measurement outcomes', 'Probability basics'],
                'experiments': ['Fixed angles (0°, 90°)', 'Single basis comparisons'],
                'challenges': ['Predict simple outcomes', 'Identify basis effects']
            },
            'intermediate': {
                'concepts': ['Superposition states', 'Basis transformations', 'Quantum uncertainty'],
                'experiments': ['Variable angles', 'Basis combinations', 'Error analysis'],
                'challenges': ['Calculate exact probabilities', 'Design optimal measurements']
            },
            'advanced': {
                'concepts': ['Quantum information theory', 'Security analysis', 'Protocol design'],
                'experiments': ['Realistic noise models', 'Security protocols', 'Custom strategies'],
                'challenges': ['Optimize key rates', 'Analyze attack scenarios']
            }
        }
        
        # Safe access with fallback
        level = analysis.get('level', 'beginner')
        return content.get(level, content['beginner'])

    @staticmethod
    def provide_intelligent_feedback(user_answer: str, correct_answer: str, concept: str) -> str:
        """Generate intelligent, contextual feedback"""
        misconception_feedback = {
            'basis_confusion': {
                'feedback': "ðŸ” **Key Insight**: Wrong basis = Random results! Think of it like asking the wrong question.",
                'hint': "Try the same state with both bases and compare results."
            },
            'probability_error': {
                'feedback': "ðŸ“ **Math Check**: Remember Born's rule: P = |amplitude|Â². The angle matters!",
                'hint': "For Î¸ degrees: P(|0âŸ©) = cosÂ²(Î¸/2), P(|1âŸ©) = sinÂ²(Î¸/2)"
            },
            'superposition_misunderstanding': {
                'feedback': "âš›ï¸ **Quantum Magic**: The photon is BOTH |0âŸ© AND |1âŸ© simultaneously before measurement!",
                'hint': "Superposition â‰  classical mixture. It's genuinely quantum."
            }
        }
        
        # Simple error detection logic
        if 'basis' in concept and user_answer != correct_answer:
            return misconception_feedback['basis_confusion']['feedback']
        elif 'probability' in concept:
            return misconception_feedback['probability_error']['feedback']
        else:
            return misconception_feedback['superposition_misunderstanding']['feedback']
    
    # =============================================================================
    # ENHANCED VISUALIZATION FUNCTIONS
    # =============================================================================
    
    def create_bloch_sphere_visualization(self, theta: float) -> plt.Figure:
        """Create interactive Bloch sphere with quantum state vector"""
        fig = plt.figure(figsize=(12, 5))
        
        # Create subplot layout
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        
        # Bloch sphere visualization
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Draw transparent sphere
        ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')
        
        # Draw axes
        ax1.quiver(0, 0, 0, 1, 0, 0, color='red', arrow_length_ratio=0.1, label='X')
        ax1.quiver(0, 0, 0, 0, 1, 0, color='green', arrow_length_ratio=0.1, label='Y')
        ax1.quiver(0, 0, 0, 0, 0, 1, color='blue', arrow_length_ratio=0.1, label='Z')
        
        # Calculate state vector position
        theta_rad = np.pi * theta / 180
        x = np.sin(theta_rad)
        y = 0
        z = np.cos(theta_rad)
        
        # Draw state vector
        ax1.quiver(0, 0, 0, x, y, z, color='purple', arrow_length_ratio=0.1, linewidth=3)
        ax1.text(x*1.2, y*1.2, z*1.2, f'|ÏˆâŸ©\n({theta}Â°)', fontsize=12, fontweight='bold')
        
        # Label poles
        ax1.text(0, 0, 1.2, '|0âŸ©', fontsize=14, ha='center', fontweight='bold')
        ax1.text(0, 0, -1.2, '|1âŸ©', fontsize=14, ha='center', fontweight='bold')
        ax1.text(1.2, 0, 0, '|+âŸ©', fontsize=14, ha='center', fontweight='bold')
        ax1.text(-1.2, 0, 0, '|-âŸ©', fontsize=14, ha='center', fontweight='bold')
        
        ax1.set_title('Quantum State on Bloch Sphere')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Probability bar chart
        prob_0 = np.cos(theta_rad / 2) ** 2
        prob_1 = np.sin(theta_rad / 2) ** 2
        
        states = ['|0âŸ©\n(Vertical)', '|1âŸ©\n(Horizontal)']
        probabilities = [prob_0, prob_1]
        colors = ['lightblue', 'lightcoral']
        
        bars = ax2.bar(states, probabilities, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Measurement Probability', fontsize=12)
        ax2.set_title(f'Measurement Probabilities at {theta}Â°', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def create_animated_bloch_sphere(self, start_theta: float, end_theta: float, 
                                   steps: int = 20, speed: float = 0.2) -> None:
        """Create animated transition between quantum states"""
        angles = np.linspace(start_theta, end_theta, steps)
        
        placeholder = st.empty()
        
        for i, theta in enumerate(angles):
            with placeholder.container():
                fig = self.create_bloch_sphere_visualization(theta)
                st.pyplot(fig)
                
                # Progress indicator
                st.progress((i + 1) / steps)
                time.sleep(speed)
        
        st.success(f"âœ¨ Animation complete! State evolved from {start_theta}Â° to {end_theta}Â°")
    
    def create_interactive_bloch_sphere(self, theta: float, phi: float, show_projections: bool = True) -> go.Figure:
        """Create interactive 3D Bloch sphere using Plotly"""
        # Sphere coordinates
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig = go.Figure()
        
        # Add transparent sphere
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.15,
            colorscale='Blues',
            showscale=False,
            hovertemplate="Bloch Sphere<extra></extra>"
        ))
        
        # Calculate state vector
        theta_rad = np.pi * theta / 180
        phi_rad = np.pi * phi / 180
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)
        
        # Add state vector
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            line=dict(color='red', width=8),
            marker=dict(size=[5, 15], color=['red', 'red']),
            name='State Vector',
            hovertemplate=f"State: Î¸={theta}Â°, Ï†={phi}Â°<br>Position: ({x:.2f}, {y:.2f}, {z:.2f})<extra></extra>"
        ))
        
        # Add projections if requested
        if show_projections:
            # Projection to XY plane
            fig.add_trace(go.Scatter3d(
                x=[x, x], y=[y, y], z=[z, 0],
                mode='lines',
                line=dict(color='gray', width=2, dash='dot'),
                name='Z Projection',
                hoverinfo='skip'
            ))
        
        # Add basis labels
        labels = [
            dict(x=0, y=0, z=1.2, text='|0âŸ©', color='blue'),
            dict(x=0, y=0, z=-1.2, text='|1âŸ©', color='red'),
            dict(x=1.2, y=0, z=0, text='|+âŸ©', color='green'),
            dict(x=-1.2, y=0, z=0, text='|-âŸ©', color='orange')
        ]
        
        for label in labels:
            fig.add_trace(go.Scatter3d(
                x=[label['x']], y=[label['y']], z=[label['z']],
                mode='text',
                text=[label['text']],
                textfont=dict(size=16, color=label['color']),
                showlegend=False,
                hovertemplate=f"{label['text']} state<extra></extra>"
            ))
        
        fig.update_layout(
            title=f"Interactive Bloch Sphere - Î¸={theta}Â°, Ï†={phi}Â°",
            scene=dict(
                xaxis=dict(range=[-1.5, 1.5], title='X'),
                yaxis=dict(range=[-1.5, 1.5], title='Y'),
                zaxis=dict(range=[-1.5, 1.5], title='Z'),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_comparative_visualization(self, theta: float, basis1: str, basis2: str) -> go.Figure:
        """Create side-by-side comparison of different measurement bases"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'{basis1} Basis', f'{basis2} Basis'],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Calculate probabilities for both bases
        prob1_0 = self.calculate_probability(theta, basis1, 0)
        prob1_1 = self.calculate_probability(theta, basis1, 1)
        prob2_0 = self.calculate_probability(theta, basis2, 0)
        prob2_1 = self.calculate_probability(theta, basis2, 1)
        
        # Add bars for first basis
        fig.add_trace(go.Bar(
            x=['|0âŸ©', '|1âŸ©'], y=[prob1_0, prob1_1],
            name=basis1, marker_color='lightblue',
            text=[f'{prob1_0:.1%}', f'{prob1_1:.1%}'],
            textposition='auto'
        ), row=1, col=1)
        
        # Add bars for second basis
        fig.add_trace(go.Bar(
            x=['|0âŸ©', '|1âŸ©'], y=[prob2_0, prob2_1],
            name=basis2, marker_color='lightcoral',
            text=[f'{prob2_0:.1%}', f'{prob2_1:.1%}'],
            textposition='auto'
        ), row=1, col=2)
        
        fig.update_layout(
            title=f"Basis Comparison at Î¸ = {theta}Â°",
            showlegend=False,
            height=400
        )
        
        fig.update_yaxes(title_text="Probability", range=[0, 1])
        
        return fig
    
    # =============================================================================
    # QUANTUM CIRCUIT AND SIMULATION FUNCTIONS
    # =============================================================================
    
    def create_enhanced_quantum_circuit(self, theta: float, basis: str, noise_params: Dict = None) -> QuantumCircuit:
        """Create quantum circuit with optional noise modeling"""
        qc = QuantumCircuit(1, 1)
        
        # State preparation
        if theta != 0:
            qc.ry(2 * np.pi * theta / 180, 0)
        
        # Add noise if specified
        if noise_params and noise_params.get('depolarizing_error', 0) > 0:
            # In real implementation, add noise gates
            pass
        
        # Measurement basis rotation
        if basis.lower() in ['diagonal', 'x']:
            qc.h(0)
        elif basis.lower() in ['circular', 'y']:
            qc.sdg(0)
            qc.h(0)
        
        qc.measure(0, 0)
        
        return qc
    
    def run_enhanced_simulation(self, theta: float, phi: float, basis: str, shots: int, learning_analysis: Dict):
        """Execute enhanced quantum simulation with learning analytics"""
        
        try:
            with st.spinner("Running enhanced quantum simulation..."):
                
                # Create and run quantum circuit
                qc = self.create_enhanced_quantum_circuit(theta, basis)
                qc_transpiled = transpile(qc, self.backend)
                
                job = self.backend.run(qc_transpiled, shots=shots)
                result = job.result()
                counts = result.get_counts()
                
                # Calculate results
                total_shots = sum(counts.values())
                prob_0_experimental = counts.get('0', 0) / total_shots
                prob_1_experimental = counts.get('1', 0) / total_shots
                
                # Calculate theoretical probabilities
                prob_0_theoretical = self.calculate_probability(theta, basis, 0)
                prob_1_theoretical = self.calculate_probability(theta, basis, 1)
                
                # Calculate accuracy
                accuracy = 1 - abs(prob_0_experimental - prob_0_theoretical)
                
                # Display results
                self.display_enhanced_results({
                    'experimental': {'0': prob_0_experimental, '1': prob_1_experimental},
                    'theoretical': {'0': prob_0_theoretical, '1': prob_1_theoretical},
                    'accuracy': accuracy,
                    'shots': shots,
                    'theta': theta,
                    'phi': phi,
                    'basis': basis
                })
                
                # Update learning analytics
                performance_record = {
                    'accuracy': accuracy,
                    'concept': f'{basis.lower()}_basis',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'parameters': {'theta': theta, 'phi': phi, 'basis': basis}
                }
                
                if 'polarization_performance' not in st.session_state:
                    st.session_state.polarization_performance = []
                
                st.session_state.polarization_performance.append(performance_record)
                
                # Provide intelligent feedback
                feedback = self.provide_intelligent_feedback("", "", f'{basis.lower()}_basis')
                st.info(feedback)
                
                # Show visualization
                self.render_enhanced_visualization(theta, phi, basis)
                
        except Exception as e:
            st.error(f"Enhanced simulation failed: {str(e)}")
    
    def run_circuit_simulation(self, theta: float, basis: str, noise_params: Dict, shots: int):
        """Run quantum circuit simulation with noise"""
        
        qc = self.create_enhanced_quantum_circuit(theta, basis, noise_params)
        results = self.run_enhanced_quantum_simulation(qc, shots)
        
        # Display circuit results
        st.subheader("ðŸŽ¯ Circuit Simulation Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Shots", results['total_shots'])
        
        with col2:
            st.metric("Entropy", f"{results['entropy']:.3f}")
        
        with col3:
            uncertainty = np.mean(list(results['uncertainties'].values()))
            st.metric("Avg Uncertainty", f"{uncertainty:.3f}")
        
        # Show probability comparison
        fig = go.Figure()
        
        states = ['|0âŸ©', '|1âŸ©']
        experimental = [results['probabilities']['0'], results['probabilities']['1']]
        
        fig.add_trace(go.Bar(
            x=states, y=experimental,
            name='Experimental',
            marker_color='lightblue',
            error_y=dict(
                type='data',
                array=[results['uncertainties']['0'], results['uncertainties']['1']],
                visible=True
            )
        ))
        
        fig.update_layout(
            title='Circuit Simulation Results with Error Bars',
            yaxis_title='Probability',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run_enhanced_quantum_simulation(self, qc: QuantumCircuit, shots: int = 1000) -> Dict:
        """Run quantum simulation with enhanced result analysis"""
        
        qc_transpiled = transpile(qc, self.backend)
        job = self.backend.run(qc_transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate additional statistics
        total_shots = sum(counts.values())
        prob_0 = counts.get('0', 0) / total_shots
        prob_1 = counts.get('1', 0) / total_shots
        
        # Calculate uncertainty
        uncertainty_0 = np.sqrt(prob_0 * (1 - prob_0) / total_shots)
        uncertainty_1 = np.sqrt(prob_1 * (1 - prob_1) / total_shots)
        
        return {
            'counts': counts,
            'probabilities': {'0': prob_0, '1': prob_1},
            'uncertainties': {'0': uncertainty_0, '1': uncertainty_1},
            'total_shots': total_shots,
            'entropy': -prob_0 * np.log2(prob_0 + 1e-10) - prob_1 * np.log2(prob_1 + 1e-10)
        }
    
    # =============================================================================
    # PROBABILITY CALCULATION FUNCTIONS
    # =============================================================================
    
    def calculate_probability(self, theta: float, basis: str, outcome: int) -> float:
        """Enhanced probability calculation with support for multiple bases"""
        theta_rad = np.pi * theta / 180
        
        basis_lower = basis.lower()
        
        if basis_lower in ['rectilinear', 'z', 'computational']:
            if outcome == 0:
                return np.cos(theta_rad / 2) ** 2
            else:
                return np.sin(theta_rad / 2) ** 2
        
        elif basis_lower in ['diagonal', 'x', 'hadamard']:
            if outcome == 0:  # |+âŸ© outcome
                return 0.5 * (1 + np.sin(theta_rad))
            else:  # |-âŸ© outcome
                return 0.5 * (1 - np.sin(theta_rad))
        
        elif basis_lower in ['circular', 'y']:
            if outcome == 0:  # |RâŸ© (right circular) outcome
                return 0.5 * (1 + np.cos(theta_rad))
            else:  # |LâŸ© (left circular) outcome
                return 0.5 * (1 - np.cos(theta_rad))
        
        else:
            raise ValueError(f"Unknown basis: {basis}")
        
    @st.cache_data
    def calculate_theoretical_statistics(self, theta: float, basis: str, shots: int) -> Dict:
        """Calculate theoretical statistics for comparison"""
        prob_0 = self.calculate_probability(theta, basis, 0)
        prob_1 = self.calculate_probability(theta, basis, 1)
        
        # Expected counts
        expected_0 = prob_0 * shots
        expected_1 = prob_1 * shots
        
        # Standard deviations
        std_0 = np.sqrt(shots * prob_0 * (1 - prob_0))
        std_1 = np.sqrt(shots * prob_1 * (1 - prob_1))
        
        return {
            'probabilities': {'0': prob_0, '1': prob_1},
            'expected_counts': {'0': expected_0, '1': expected_1},
            'standard_deviations': {'0': std_0, '1': std_1},
            'entropy': -prob_0 * np.log2(prob_0 + 1e-10) - prob_1 * np.log2(prob_1 + 1e-10)
        }
    
    # =============================================================================
    # HELPER FUNCTIONS
    # =============================================================================
    
    def display_enhanced_results(self, results: Dict):
        """Display enhanced simulation results"""
        
        st.subheader("ðŸŽ¯ Enhanced Simulation Results")
        
        # Create comparison chart
        fig = go.Figure()
        
        states = ['|0âŸ©', '|1âŸ©']
        experimental = [results['experimental']['0'], results['experimental']['1']]
        theoretical = [results['theoretical']['0'], results['theoretical']['1']]
        
        fig.add_trace(go.Bar(
            x=states, y=experimental,
            name='Experimental',
            marker_color='lightblue',
            text=[f'{p:.1%}' for p in experimental],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            x=states, y=theoretical,
            name='Theoretical',
            marker_color='lightcoral',
            text=[f'{p:.1%}' for p in theoretical],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Experimental vs Theoretical - {results['basis']} Basis",
            yaxis_title='Probability',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Accuracy analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.1%}")
        
        with col2:
            if results['accuracy'] > 0.9:
                st.success("ðŸŽ¯ Excellent match!")
            elif results['accuracy'] > 0.8:
                st.info("ðŸ‘ Good agreement")
            else:
                st.warning("ðŸ“Š Consider more shots")
        
        with col3:
            st.metric("Shots Used", results['shots'])
    
    def display_state_information(self, theta: float, phi: float):
        """Display detailed state information"""
        
        st.subheader("ðŸ“Š Quantum State Information")
        
        # Calculate state components
        theta_rad = np.pi * theta / 180
        phi_rad = np.pi * phi / 180
        
        alpha = np.cos(theta_rad / 2)
        beta = np.exp(1j * phi_rad) * np.sin(theta_rad / 2)
        
        # Bloch vector coordinates
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**State Vector:**")
            st.latex(f"|\\psi\\rangle = {alpha:.3f}|0\\rangle + {beta:.3f}|1\\rangle")
            
            st.write("**Bloch Coordinates:**")
            st.write(f"â€¢ X: {x:.3f}")
            st.write(f"â€¢ Y: {y:.3f}")
            st.write(f"â€¢ Z: {z:.3f}")
        
        with col2:
            st.write("**Measurement Probabilities:**")
            prob_0 = abs(alpha) ** 2
            prob_1 = abs(beta) ** 2
            
            st.write(f"â€¢ P(|0âŸ©) = {prob_0:.3f}")
            st.write(f"â€¢ P(|1âŸ©) = {prob_1:.3f}")
            
            st.write("**Phase Information:**")
            phase = np.angle(beta)
            st.write(f"â€¢ Ï† = {np.degrees(phase):.1f}Â°")
    
    def display_comparative_analysis(self, theta: float, basis1: str, basis2: str):
        """Display detailed comparative analysis"""
        
        st.subheader(f"ðŸ” Detailed Analysis at Î¸ = {theta}Â°")
        
        # Calculate probabilities for both bases
        prob1_0 = self.calculate_probability(theta, basis1, 0)
        prob1_1 = self.calculate_probability(theta, basis1, 1)
        prob2_0 = self.calculate_probability(theta, basis2, 0)
        prob2_1 = self.calculate_probability(theta, basis2, 1)
        
        # Analysis table
        analysis_data = {
            'Basis': [basis1, basis2],
            'P(|0âŸ©)': [f"{prob1_0:.3f}", f"{prob2_0:.3f}"],
            'P(|1âŸ©)': [f"{prob1_1:.3f}", f"{prob2_1:.3f}"],
            'Uncertainty': [f"{prob1_0 * prob1_1:.3f}", f"{prob2_0 * prob2_1:.3f}"],
            'Entropy': [
                f"{-prob1_0 * np.log2(prob1_0 + 1e-10) - prob1_1 * np.log2(prob1_1 + 1e-10):.3f}",
                f"{-prob2_0 * np.log2(prob2_0 + 1e-10) - prob2_1 * np.log2(prob2_1 + 1e-10):.3f}"
            ]
        }
        
        df = pd.DataFrame(analysis_data)
        st.dataframe(df, use_container_width=True)
        
        # Key insights
        st.write("**ðŸ” Key Insights:**")
        
        if abs(prob1_0 - prob2_0) < 0.1:
            st.write("â€¢ Probabilities are similar between bases")
        else:
            st.write("â€¢ Significant difference in measurement outcomes between bases")
        
        if prob1_0 * prob1_1 > prob2_0 * prob2_1:
            st.write(f"â€¢ {basis1} basis shows higher quantum uncertainty")
        else:
            st.write(f"â€¢ {basis2} basis shows higher quantum uncertainty")
    
    def render_enhanced_visualization(self, theta: float, phi: float, basis: str):
        """Render enhanced visualization combining multiple views"""
        
        st.subheader("ðŸŒŸ Enhanced Quantum Visualization")
        
        # Create combined visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Bloch Sphere View',
                'Probability Distribution',
                'Phase Representation',
                'Uncertainty Analysis'
            ],
            specs=[
                [{"type": "scatter3d"}, {"type": "bar"}],
                [{"type": "polar"}, {"type": "scatter"}]
            ]
        )
        
        # Bloch sphere (simplified 2D projection)
        theta_rad = np.pi * theta / 180
        phi_rad = np.pi * phi / 180
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)
        
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            marker=dict(size=[5, 10], color=['blue', 'red']),
            line=dict(width=5),
            name='State Vector'
        ), row=1, col=1)
        
        # Probability distribution
        prob_0 = self.calculate_probability(theta, basis, 0)
        prob_1 = self.calculate_probability(theta, basis, 1)
        
        fig.add_trace(go.Bar(
            x=['|0âŸ©', '|1âŸ©'], y=[prob_0, prob_1],
            marker_color=['lightblue', 'lightcoral'],
            name='Probabilities',
            text=[f'{prob_0:.1%}', f'{prob_1:.1%}'],
            textposition='auto'
        ), row=1, col=2)
        
        # Phase representation
        fig.add_trace(go.Scatterpolar(
            r=[1], theta=[np.degrees(phi_rad)],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Phase'
        ), row=2, col=1)
        
        # Uncertainty analysis
        angles = np.linspace(0, 180, 50)
        uncertainties = [
            self.calculate_probability(angle, basis, 0) * self.calculate_probability(angle, basis, 1)
            for angle in angles
        ]
        
        fig.add_trace(go.Scatter(
            x=angles, y=uncertainties,
            mode='lines',
            line=dict(color='green', width=3),
            name='Uncertainty'
        ), row=2, col=2)
        
        # Add current angle marker
        current_uncertainty = prob_0 * prob_1
        fig.add_trace(go.Scatter(
            x=[theta], y=[current_uncertainty],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Current State'
        ), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Enhanced Quantum State Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    def calculate_improvement_trend(self, user_performance: List[Dict]) -> float:
        """Calculate user's improvement trend"""
        if len(user_performance) < 2:
            return 0.0
        
        accuracies = [p['accuracy'] for p in user_performance]
        
        # Simple linear trend
        x = np.arange(len(accuracies))
        coeffs = np.polyfit(x, accuracies, 1)
        
        return coeffs[0] * len(accuracies)  # Slope * length gives overall improvement
    
    def create_performance_dashboard(self, user_performance: List[Dict]):
        """Create performance analytics dashboard"""
        
        if len(user_performance) < 2:
            st.info("Complete more experiments to see performance trends!")
            return
        
        # Extract data
        timestamps = [p.get('timestamp', f"Experiment {i+1}") for i, p in enumerate(user_performance)]
        accuracies = [p['accuracy'] for p in user_performance]
        concepts = [p.get('concept', 'general') for p in user_performance]
        
        # Performance over time
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Accuracy Over Time',
                'Performance by Concept',
                'Learning Curve Fit',
                'Recent Performance'
            ]
        )
        
        # Accuracy trend
        fig.add_trace(go.Scatter(
            x=list(range(len(accuracies))),
            y=accuracies,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='blue', width=3)
        ), row=1, col=1)
        
        # Performance by concept
        concept_performance = {}
        for concept, accuracy in zip(concepts, accuracies):
            if concept not in concept_performance:
                concept_performance[concept] = []
            concept_performance[concept].append(accuracy)
        
        concept_avg = {k: np.mean(v) for k, v in concept_performance.items()}
        
        fig.add_trace(go.Bar(
            x=list(concept_avg.keys()),
            y=list(concept_avg.values()),
            marker_color='lightgreen',
            name='Avg Accuracy'
        ), row=1, col=2)
        
        # Learning curve fit
        x = np.arange(len(accuracies))
        if len(accuracies) > 2:
            coeffs = np.polyfit(x, accuracies, min(2, len(accuracies)-1))
            trend = np.polyval(coeffs, x)
            
            fig.add_trace(go.Scatter(
                x=x, y=accuracies,
                mode='markers',
                name='Actual',
                marker=dict(color='blue')
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=x, y=trend,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=3)
            ), row=2, col=1)
        
        # Recent performance (last 5)
        recent_accuracies = accuracies[-5:]
        recent_x = list(range(len(recent_accuracies)))
        
        fig.add_trace(go.Bar(
            x=recent_x,
            y=recent_accuracies,
            marker_color='orange',
            name='Recent'
        ), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Learning Analytics Dashboard")
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_ai_recommendations(self, analysis: Dict, user_performance: List[Dict]) -> List[str]:
        """Generate AI-powered learning recommendations"""
        
        recommendations = []
        
        # Level-based recommendations
        if analysis['level'] == 'beginner':
            recommendations.extend([
                "ðŸŽ¯ Focus on understanding basic polarization concepts",
                "ðŸ“š Practice with simple angles (0Â°, 45Â°, 90Â°, 180Â°)",
                "ðŸ” Explore the relationship between angle and probability"
            ])
        elif analysis['level'] == 'intermediate':
            recommendations.extend([
                "ðŸ§® Practice calculating exact probabilities",
                "ðŸ”„ Compare different measurement bases",
                "ðŸ“Š Analyze quantum uncertainty principles"
            ])
        else:
            recommendations.extend([
                "ðŸš€ Explore advanced quantum protocols",
                "ðŸ”¬ Study realistic noise models",
                "ðŸ“ˆ Optimize measurement strategies"
            ])
        
        # Performance-based recommendations
        if analysis['recent_accuracy'] < 0.7:
            recommendations.append("ðŸ“– Review fundamental concepts before advancing")
        elif analysis['recent_accuracy'] > 0.9:
            recommendations.append("ðŸŽ–ï¸ Excellent! Consider exploring advanced topics")
        
        # Focus area recommendations
        if 'basis_choice' in analysis['focus_areas']:
            recommendations.append("ðŸŽ¯ Practice basis selection strategies")
        
        if 'polarization' in analysis['focus_areas']:
            recommendations.append("ðŸ”¬ Focus on polarization angle relationships")
        
        return recommendations[:5]  # Return top 5 recommendations


def main():
    EnhancedPolarizationModule().main()

if __name__ == "__main__":
    main()