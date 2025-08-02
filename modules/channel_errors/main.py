"""
Channel Errors & Noise Simulation Module
Teaches quantum decoherence and noise modeling
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from typing import Dict, Any, List, Tuple
from core.session_manager import SessionManager
from utils.visualizations import plot_probabilities, plot_error_analysis
from components.ui_components import ParameterControls, ResultsDisplay

class ChannelErrorsModule:
    """Main channel errors simulation module"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.backend = AerSimulator()
    
    def main(self):
        """Main interface for channel errors module"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff7b7b 0%, #d63384 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.2em;">📡 Channel Errors & Noise</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1em;">Understanding Quantum Decoherence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Learning tabs
        tabs = st.tabs([
            "🧪 Noise Laboratory", 
            "📊 Error Analysis", 
            "🎓 Educational Content",
            "📈 Advanced Studies"
        ])
        
        with tabs[0]:
            self.render_noise_laboratory()
        
        with tabs[1]:
            self.render_error_analysis()
        
        with tabs[2]:
            self.render_educational_content()
        
        with tabs[3]:
            self.render_advanced_studies()
    
    def render_noise_laboratory(self):
        """Interactive noise simulation laboratory"""
        st.header("🧪 Quantum Noise Laboratory")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎛️ Noise Parameters")
            
            # Noise type selection
            noise_type = st.selectbox(
                "Noise Model",
                ["Depolarization", "Amplitude Damping", "Phase Damping", "Bit Flip", "Custom Mix"]
            )
            
            # Noise strength
            noise_strength = st.slider("Noise Strength", 0.0, 0.5, 0.1, 0.01)
            
            # Initial state preparation
            st.subheader("🌐 Initial State")
            initial_theta = st.slider("Initial θ", 0, 180, 45)
            initial_phi = st.slider("Initial φ", 0, 360, 0)
            
            # Simulation parameters
            shots = st.select_slider("Shots", [100, 500, 1000, 5000], value=1000)
            
            if st.button("🚀 Run Noise Simulation", type="primary"):
                self.run_noise_simulation(noise_type, noise_strength, initial_theta, initial_phi, shots)
        
        with col2:
            self.render_noise_visualization()
    
    def run_noise_simulation(self, noise_type: str, strength: float, theta: float, phi: float, shots: int):
        """Execute noise simulation"""
        
        # Create quantum circuit
        qc = self.create_state_circuit(theta, phi)
        
        # Create noise model
        noise_model = self.create_noise_model(noise_type, strength)
        
        # Run simulation with and without noise
        clean_results = self.run_circuit(qc, shots)
        noisy_results = self.run_circuit(qc, shots, noise_model)
        
        # Display results
        self.display_noise_results(clean_results, noisy_results, noise_type, strength)
        
        # Update session
        self.session_manager.add_experiment_record('channel_errors', {
            'noise_type': noise_type,
            'strength': strength,
            'theta': theta,
            'phi': phi,
            'clean_results': clean_results,
            'noisy_results': noisy_results
        })
    
    def create_state_circuit(self, theta: float, phi: float) -> QuantumCircuit:
        """Create quantum circuit for state preparation"""
        qc = QuantumCircuit(1, 1)
        
        # State preparation
        if theta != 0:
            qc.ry(np.pi * theta / 180, 0)
        if phi != 0:
            qc.rz(np.pi * phi / 180, 0)
        
        qc.measure(0, 0)
        return qc
    
    def create_noise_model(self, noise_type: str, strength: float) -> NoiseModel:
        """Create Qiskit noise model"""
        noise_model = NoiseModel()
        
        if noise_type == "Depolarization":
            error = depolarizing_error(strength, 1)
            noise_model.add_all_qubit_quantum_error(error, ['ry', 'rz'])
        elif noise_type == "Amplitude Damping":
            error = amplitude_damping_error(strength)
            noise_model.add_all_qubit_quantum_error(error, ['ry', 'rz'])
        elif noise_type == "Phase Damping":
            error = phase_damping_error(strength)
            noise_model.add_all_qubit_quantum_error(error, ['ry', 'rz'])
        elif noise_type == "Bit Flip":
            from qiskit_aer.noise import pauli_error
            error = pauli_error([('X', strength), ('I', 1 - strength)])
            noise_model.add_all_qubit_quantum_error(error, ['ry', 'rz'])
        
        return noise_model
    
    def run_circuit(self, qc: QuantumCircuit, shots: int, noise_model=None) -> Dict:
        """Run quantum circuit simulation"""
        transpiled_qc = transpile(qc, self.backend)
        
        if noise_model:
            job = self.backend.run(transpiled_qc, shots=shots, noise_model=noise_model)
        else:
            job = self.backend.run(transpiled_qc, shots=shots)
        
        result = job.result()
        counts = result.get_counts()
        
        # Process results
        total_shots = sum(counts.values())
        prob_0 = counts.get('0', 0) / total_shots
        prob_1 = counts.get('1', 0) / total_shots
        
        return {
            'counts': counts,
            'probabilities': {'0': prob_0, '1': prob_1},
            'total_shots': total_shots
        }
    
    def display_noise_results(self, clean: Dict, noisy: Dict, noise_type: str, strength: float):
        """Display simulation results"""
        st.subheader("📊 Simulation Results")
        
        # Metrics comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Clean P(|0⟩)",
                f"{clean['probabilities']['0']:.3f}",
                help="Probability without noise"
            )
        
        with col2:
            st.metric(
                "Noisy P(|0⟩)",
                f"{noisy['probabilities']['0']:.3f}",
                delta=f"{noisy['probabilities']['0'] - clean['probabilities']['0']:.3f}"
            )
        
        with col3:
            fidelity = self.calculate_fidelity(clean['probabilities'], noisy['probabilities'])
            st.metric("Fidelity", f"{fidelity:.3f}", help="Similarity to clean state")
        
        # Visualization
        self.create_comparison_chart(clean, noisy, noise_type)
    
    def calculate_fidelity(self, clean_probs: Dict, noisy_probs: Dict) -> float:
        """Calculate fidelity between clean and noisy states"""
        clean_0 = np.sqrt(clean_probs['0'])
        clean_1 = np.sqrt(clean_probs['1'])
        noisy_0 = np.sqrt(noisy_probs['0'])
        noisy_1 = np.sqrt(noisy_probs['1'])
        
        return (clean_0 * noisy_0 + clean_1 * noisy_1) ** 2
    
    def create_comparison_chart(self, clean: Dict, noisy: Dict, noise_type: str):
        """Create comparison visualization"""
        fig = go.Figure()
        
        states = ['|0⟩', '|1⟩']
        clean_probs = [clean['probabilities']['0'], clean['probabilities']['1']]
        noisy_probs = [noisy['probabilities']['0'], noisy['probabilities']['1']]
        
        fig.add_trace(go.Bar(
            x=states,
            y=clean_probs,
            name='Clean',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=states,
            y=noisy_probs,
            name=f'{noise_type} Noise',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f'Clean vs Noisy State Comparison',
            xaxis_title='Quantum State',
            yaxis_title='Probability',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_noise_visualization(self):
        """Render noise model visualizations"""
        st.subheader("📈 Noise Effects Visualization")
        
        # Create noise strength vs fidelity plot
        strengths = np.linspace(0, 0.5, 50)
        fidelities = 1 - strengths  # Simplified model
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=strengths,
            y=fidelities,
            mode='lines',
            name='Fidelity vs Noise',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title='Quantum State Fidelity vs Noise Strength',
            xaxis_title='Noise Strength',
            yaxis_title='Fidelity',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_error_analysis(self):
        """Error rate analysis interface"""
        st.header("📊 Quantum Error Rate Analysis")
        
        # Error rate trends
        if st.button("📈 Generate Error Trend Analysis"):
            self.generate_error_trends()
        
        # QBER analysis
        st.subheader("🎯 Quantum Bit Error Rate (QBER)")
        with st.expander("QBER Calculator"):
            sent_bits = st.number_input("Bits Sent", min_value=100, max_value=10000, value=1000)
            error_bits = st.number_input("Error Bits", min_value=0, max_value=sent_bits, value=50)
            
            if sent_bits > 0:
                qber = error_bits / sent_bits
                st.metric("QBER", f"{qber:.1%}")
                
                if qber < 0.11:
                    st.success("✅ Below security threshold (11%)")
                else:
                    st.error("❌ Above security threshold - communication compromised")
    
    def generate_error_trends(self):
        """Generate error trend analysis"""
        # Simulate error rates over time
        time_points = np.arange(1, 101)
        base_error = 0.05
        noise_variation = 0.02 * np.random.randn(100)
        error_rates = base_error + noise_variation + 0.001 * time_points
        error_rates = np.clip(error_rates, 0, 0.5)
        
        fig = plot_error_analysis(error_rates.tolist(), 
                                title="Error Rate Evolution Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_educational_content(self):
        """Educational content about quantum noise"""
        st.header("🎓 Understanding Quantum Noise")
        
        # Noise types explanation
        noise_tabs = st.tabs(["Depolarization", "Amplitude Damping", "Phase Damping", "Bit Flip"])
        
        with noise_tabs[0]:
            st.subheader("🌀 Depolarization Noise")
            st.markdown("""
            **Physical Origin**: Random interactions with environment
            
            **Mathematical Model**: 
            ρ → (1-p)ρ + p(I/2) where p is noise strength
            
            **Effect**: Gradually transforms any quantum state into maximally mixed state
            
            **Real-world Example**: Thermal fluctuations in quantum processors
            """)
        
        with noise_tabs[1]:
            st.subheader("📉 Amplitude Damping")
            st.markdown("""
            **Physical Origin**: Energy loss to environment (T₁ process)
            
            **Mathematical Model**: |1⟩ → √(1-γ)|1⟩ + √γ|0⟩
            
            **Effect**: Preferentially causes |1⟩ → |0⟩ transitions
            
            **Real-world Example**: Photon loss in optical fibers, qubit relaxation
            """)
        
        with noise_tabs[2]:
            st.subheader("🌊 Phase Damping")
            st.markdown("""
            **Physical Origin**: Random phase kicks from environment (T₂ process)
            
            **Mathematical Model**: Destroys off-diagonal elements of density matrix
            
            **Effect**: Preserves populations but destroys quantum coherence
            
            **Real-world Example**: Magnetic field fluctuations in spin qubits
            """)
        
        with noise_tabs[3]:
            st.subheader("🔄 Bit Flip Noise")
            st.markdown("""
            **Physical Origin**: Classical-like errors in quantum systems
            
            **Mathematical Model**: Apply Pauli-X with probability p
            
            **Effect**: Randomly flips |0⟩ ↔ |1⟩
            
            **Real-world Example**: Gate errors in quantum computers
            """)
    
    def render_advanced_studies(self):
        """Advanced noise analysis tools"""
        st.header("📈 Advanced Noise Studies")
        
        # Process tomography simulation
        if st.button("🔬 Process Tomography Simulation"):
            self.process_tomography_demo()
        
        # Noise correlations
        st.subheader("📊 Noise Correlation Analysis")
        with st.expander("Multi-qubit Noise Correlations"):
            n_qubits = st.slider("Number of Qubits", 2, 5, 2)
            correlation_strength = st.slider("Correlation Strength", 0.0, 1.0, 0.3)
            
            if st.button("Generate Correlation Matrix"):
                self.generate_noise_correlations(n_qubits, correlation_strength)
    
    def process_tomography_demo(self):
        """Demonstrate quantum process tomography"""
        st.subheader("🔬 Process Tomography Results")
        
        # Simulate process matrix
        ideal_process = np.array([[1, 0, 0, 1],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [1, 0, 0, 1]]) / 2
        
        noise_strength = 0.1
        noisy_process = (1 - noise_strength) * ideal_process + noise_strength * np.eye(4) / 4
        
        # Visualize process matrices
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Ideal Process', 'Noisy Process'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        fig.add_trace(go.Heatmap(z=ideal_process, colorscale='Blues'), row=1, col=1)
        fig.add_trace(go.Heatmap(z=noisy_process, colorscale='Reds'), row=1, col=2)
        
        fig.update_layout(title="Process Tomography Comparison", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_noise_correlations(self, n_qubits: int, correlation: float):
        """Generate noise correlation matrix"""
        # Create correlated noise matrix
        base_matrix = np.eye(n_qubits)
        correlation_matrix = correlation * np.ones((n_qubits, n_qubits)) + (1 - correlation) * base_matrix
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            colorscale='RdBu',
            zmid=0.5
        ))
        
        fig.update_layout(
            title=f'{n_qubits}-Qubit Noise Correlation Matrix',
            xaxis_title='Qubit Index',
            yaxis_title='Qubit Index'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function for channel errors module"""
    module = ChannelErrorsModule()
    module.main()
