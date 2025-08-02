"""
BB84 Protocol Module for Quantum Key Distribution
Implements the original quantum cryptography protocol
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from typing import Dict, Any, List, Tuple
from core.session_manager import SessionManager

class BB84Module:
    """Main BB84 protocol simulation module"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.backend = AerSimulator()
    
    def main(self):
        """Main interface for BB84 module"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.2em;">🔐 BB84 Protocol</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1em;">The Original Quantum Key Distribution Protocol</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main tabs
        tabs = st.tabs([
            "🎓 Tutorial", 
            "🧪 Interactive Simulation", 
            "📊 Analysis",
            "🔬 Advanced Features"
        ])
        
        with tabs[0]:
            self.render_tutorial()
        
        with tabs[1]:
            self.render_simulation()
        
        with tabs[2]:
            self.render_analysis()
        
        with tabs[3]:
            self.render_advanced_features()
    
    def render_tutorial(self):
        """Render BB84 protocol tutorial"""
        st.header("🎓 BB84 Protocol Tutorial")
        
        st.markdown("""
        ### What is BB84?
        
        BB84 is the first quantum key distribution protocol, invented by Charles Bennett and Gilles Brassard in 1984. 
        It allows two parties (Alice and Bob) to establish a shared cryptographic key with information-theoretic security.
        
        ### Protocol Steps:
        
        1. **🎲 Random Generation**: Alice generates random bits and bases
        2. **📡 Quantum Transmission**: Alice sends qubits to Bob using chosen bases
        3. **📐 Measurement**: Bob randomly chooses bases and measures received qubits
        4. **📢 Basis Sifting**: Alice and Bob publicly compare bases, keep matching ones
        5. **🔍 Error Detection**: Sample subset to check for eavesdropping
        6. **🔧 Error Correction**: Correct remaining errors in the key
        7. **🔒 Privacy Amplification**: Extract final secure key
        """)
        
        # Interactive protocol demonstration
        st.subheader("🎮 Step-by-Step Demonstration")
        
        demo_step = st.selectbox("Select Protocol Step", [
            "1. Bit & Basis Generation",
            "2. Quantum State Preparation", 
            "3. Transmission & Measurement",
            "4. Basis Sifting",
            "5. Error Detection"
        ])
        
        if demo_step == "1. Bit & Basis Generation":
            self.demo_bit_generation()
        elif demo_step == "2. Quantum State Preparation":
            self.demo_state_preparation()
        elif demo_step == "3. Transmission & Measurement":
            self.demo_transmission()
        elif demo_step == "4. Basis Sifting":
            self.demo_basis_sifting()
        elif demo_step == "5. Error Detection":
            self.demo_error_detection()
    
    def demo_bit_generation(self):
        """Demonstrate random bit and basis generation"""
        st.write("**Step 1: Alice generates random bits and bases**")
        
        n_demo = st.slider("Number of bits to generate", 5, 20, 8)
        
        if st.button("🎲 Generate Random Bits & Bases"):
            # Generate random bits and bases
            alice_bits = np.random.randint(0, 2, n_demo)
            alice_bases = np.random.randint(0, 2, n_demo)  # 0: rectilinear, 1: diagonal
            
            # Display in table format
            import pandas as pd
            df = pd.DataFrame({
                'Position': range(n_demo),
                'Alice Bit': alice_bits,
                'Alice Basis': ['↔' if b == 0 else '⤢' for b in alice_bases],
                'Quantum State': [self.get_state_notation(bit, basis) for bit, basis in zip(alice_bits, alice_bases)]
            })
            
            st.dataframe(df, use_container_width=True)
            
            # Store in session for next steps
            st.session_state['demo_alice_bits'] = alice_bits
            st.session_state['demo_alice_bases'] = alice_bases
    
    def demo_state_preparation(self):
        """Demonstrate quantum state preparation"""
        st.write("**Step 2: Alice prepares quantum states based on bits and bases**")
        
        if 'demo_alice_bits' not in st.session_state:
            st.info("Please complete Step 1 first!")
            return
        
        alice_bits = st.session_state['demo_alice_bits']
        alice_bases = st.session_state['demo_alice_bases']
        
        st.write("**Encoding Rules:**")
        st.write("- Rectilinear basis (↔): |0⟩ for bit 0, |1⟩ for bit 1")
        st.write("- Diagonal basis (⤢): |+⟩ for bit 0, |-⟩ for bit 1")
        
        # Show quantum circuits for each state
        for i in range(len(alice_bits)):
            qc = self.create_bb84_state_circuit(alice_bits[i], alice_bases[i])
            st.write(f"**Position {i}:** {self.get_state_notation(alice_bits[i], alice_bases[i])}")
    
    def demo_transmission(self):
        """Demonstrate transmission and measurement"""
        st.write("**Step 3: Bob randomly chooses bases and measures**")
        
        if 'demo_alice_bits' not in st.session_state:
            st.info("Please complete Step 1 first!")
            return
        
        alice_bits = st.session_state['demo_alice_bits']
        alice_bases = st.session_state['demo_alice_bases']
        
        if st.button("🎯 Bob Measures"):
            # Bob chooses random bases
            bob_bases = np.random.randint(0, 2, len(alice_bits))
            
            # Simulate measurements
            bob_results = []
            for i in range(len(alice_bits)):
                if alice_bases[i] == bob_bases[i]:
                    # Same basis - Bob gets Alice's bit
                    bob_results.append(alice_bits[i])
                else:
                    # Different basis - random result
                    bob_results.append(np.random.randint(0, 2))
            
            # Display results
            import pandas as pd
            df = pd.DataFrame({
                'Position': range(len(alice_bits)),
                'Alice Sent': [self.get_state_notation(alice_bits[i], alice_bases[i]) for i in range(len(alice_bits))],
                'Bob Basis': ['↔' if b == 0 else '⤢' for b in bob_bases],
                'Bob Result': bob_results,
                'Bases Match': ['✅' if alice_bases[i] == bob_bases[i] else '❌' for i in range(len(alice_bits))]
            })
            
            st.dataframe(df, use_container_width=True)
            
            # Store results
            st.session_state['demo_bob_bases'] = bob_bases
            st.session_state['demo_bob_results'] = bob_results
    
    def demo_basis_sifting(self):
        """Demonstrate basis sifting process"""
        st.write("**Step 4: Alice and Bob compare bases publicly**")
        
        if 'demo_bob_bases' not in st.session_state:
            st.info("Please complete Step 3 first!")
            return
        
        alice_bits = st.session_state['demo_alice_bits']
        alice_bases = st.session_state['demo_alice_bases']
        bob_bases = st.session_state['demo_bob_bases']
        bob_results = st.session_state['demo_bob_results']
        
        # Find matching bases
        matching_bases = alice_bases == bob_bases
        sifted_alice = alice_bits[matching_bases]
        sifted_bob = np.array(bob_results)[matching_bases]
        
        st.write(f"**Sifted key length:** {len(sifted_alice)} bits")
        
        if len(sifted_alice) > 0:
            import pandas as pd
            df = pd.DataFrame({
                'Position': np.where(matching_bases)[0],
                'Alice Bit': sifted_alice,
                'Bob Bit': sifted_bob,
                'Match': ['✅' if sifted_alice[i] == sifted_bob[i] else '❌' for i in range(len(sifted_alice))]
            })
            
            st.dataframe(df, use_container_width=True)
            
            error_rate = np.sum(sifted_alice != sifted_bob) / len(sifted_alice)
            st.metric("Error Rate", f"{error_rate:.1%}")
    
    def demo_error_detection(self):
        """Demonstrate error detection"""
        st.write("**Step 5: Error rate analysis for eavesdropping detection**")
        
        st.markdown("""
        ### Security Analysis
        
        - **Low error rate (< 11%)**: Channel appears secure
        - **High error rate (> 11%)**: Possible eavesdropping detected
        - **Very high error rate (> 25%)**: Definite security breach
        
        The 11% threshold comes from information theory - above this rate, 
        no secure key can be extracted even with perfect error correction.
        """)
    
    def render_simulation(self):
        """Render interactive BB84 simulation"""
        st.header("🧪 Interactive BB84 Simulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Simulation Parameters")
            
            n_bits = st.slider("Number of Qubits", 20, 200, 100)
            noise_level = st.slider("Channel Noise (%)", 0, 30, 5)
            eavesdropper = st.checkbox("Include Eavesdropper (Eve)")
            
            if eavesdropper:
                eve_intercept_rate = st.slider("Eve Intercept Rate (%)", 10, 100, 50)
            else:
                eve_intercept_rate = 0
            
            if st.button("🚀 Run BB84 Simulation", type="primary"):
                self.run_bb84_simulation(n_bits, noise_level / 100, eve_intercept_rate / 100)
        
        with col2:
            self.render_protocol_diagram()
    
    def run_bb84_simulation(self, n_bits: int, noise_level: float, eve_rate: float):
        """Run complete BB84 simulation"""
        
        # Step 1: Alice generates random bits and bases
        alice_bits = np.random.randint(0, 2, n_bits)
        alice_bases = np.random.randint(0, 2, n_bits)
        
        # Step 2: Bob chooses random bases
        bob_bases = np.random.randint(0, 2, n_bits)
        
        # Step 3: Simulate transmission with optional eavesdropping
        if eve_rate > 0:
            bob_measurements = self.simulate_with_eavesdropping(
                alice_bits, alice_bases, bob_bases, eve_rate, noise_level
            )
        else:
            bob_measurements = self.simulate_transmission(
                alice_bits, alice_bases, bob_bases, noise_level
            )
        
        # Step 4: Basis sifting
        matching_bases = alice_bases == bob_bases
        sifted_alice = alice_bits[matching_bases]
        sifted_bob = bob_measurements[matching_bases]
        
        # Step 5: Error analysis
        if len(sifted_alice) > 0:
            errors = np.sum(sifted_alice != sifted_bob)
            error_rate = errors / len(sifted_alice)
        else:
            error_rate = 0
            errors = 0
        
        # Display results
        self.display_simulation_results({
            'n_bits': n_bits,
            'sifted_length': len(sifted_alice),
            'error_rate': error_rate,
            'errors': errors,
            'eve_present': eve_rate > 0,
            'noise_level': noise_level,
            'alice_bits': sifted_alice,
            'bob_bits': sifted_bob
        })
        
        # Update session
        self.session_manager.add_experiment_record('bb84', {
            'n_bits': n_bits,
            'noise_level': noise_level,
            'eve_rate': eve_rate,
            'error_rate': error_rate,
            'sifted_length': len(sifted_alice)
        })
    
    def simulate_transmission(self, alice_bits: np.ndarray, alice_bases: np.ndarray, 
                            bob_bases: np.ndarray, noise_level: float) -> np.ndarray:
        """Simulate quantum transmission without eavesdropping"""
        
        bob_measurements = np.zeros(len(alice_bits), dtype=int)
        
        for i in range(len(alice_bits)):
            if alice_bases[i] == bob_bases[i]:
                # Same basis - Bob should get Alice's bit (with noise)
                if np.random.random() < noise_level:
                    bob_measurements[i] = 1 - alice_bits[i]  # Flip bit due to noise
                else:
                    bob_measurements[i] = alice_bits[i]
            else:
                # Different basis - random result
                bob_measurements[i] = np.random.randint(0, 2)
        
        return bob_measurements
    
    def simulate_with_eavesdropping(self, alice_bits: np.ndarray, alice_bases: np.ndarray,
                                  bob_bases: np.ndarray, eve_rate: float, noise_level: float) -> np.ndarray:
        """Simulate transmission with eavesdropping"""
        
        bob_measurements = np.zeros(len(alice_bits), dtype=int)
        
        for i in range(len(alice_bits)):
            if np.random.random() < eve_rate:
                # Eve intercepts this qubit
                eve_basis = np.random.randint(0, 2)
                
                if alice_bases[i] == eve_basis:
                    # Eve measures in correct basis
                    eve_result = alice_bits[i]
                else:
                    # Eve measures in wrong basis - random result
                    eve_result = np.random.randint(0, 2)
                
                # Eve resends based on her measurement
                if bob_bases[i] == eve_basis:
                    bob_measurements[i] = eve_result
                else:
                    bob_measurements[i] = np.random.randint(0, 2)
            else:
                # Direct transmission (no eavesdropping)
                if alice_bases[i] == bob_bases[i]:
                    if np.random.random() < noise_level:
                        bob_measurements[i] = 1 - alice_bits[i]
                    else:
                        bob_measurements[i] = alice_bits[i]
                else:
                    bob_measurements[i] = np.random.randint(0, 2)
        
        return bob_measurements
    
    def display_simulation_results(self, results: Dict[str, Any]):
        """Display simulation results"""
        
        st.subheader("🎯 BB84 Simulation Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Bits Transmitted", results['n_bits'])
        
        with col2:
            st.metric("Sifted Key Length", results['sifted_length'])
        
        with col3:
            st.metric("Errors Detected", results['errors'])
        
        with col4:
            st.metric("Error Rate", f"{results['error_rate']:.1%}")
        
        # Security analysis
        if results['error_rate'] > 0.11:
            st.error("🚨 High error rate detected! Possible eavesdropping.")
            st.error("🛑 Abort key exchange and investigate security.")
        elif results['error_rate'] > 0.05:
            st.warning("⚠️ Moderate error rate. Monitor channel carefully.")
        else:
            st.success("✅ Low error rate. Channel appears secure.")
        
        # Efficiency metrics
        efficiency = results['sifted_length'] / results['n_bits']
        st.info(f"📊 Protocol efficiency: {efficiency:.1%} (sifted/transmitted)")
        
        # Bit comparison visualization (if small enough)
        if results['sifted_length'] <= 50:
            self.create_bit_comparison_chart(results['alice_bits'], results['bob_bits'])
    
    def create_bit_comparison_chart(self, alice_bits: np.ndarray, bob_bits: np.ndarray):
        """Create bit comparison visualization"""
        
        if len(alice_bits) == 0:
            return
        
        positions = list(range(len(alice_bits)))
        
        fig = go.Figure()
        
        # Alice's bits
        fig.add_trace(go.Scatter(
            x=positions,
            y=alice_bits,
            mode='markers+lines',
            name="Alice's Bits",
            marker=dict(color='blue', size=10),
            line=dict(color='blue', width=2)
        ))
        
        # Bob's bits
        fig.add_trace(go.Scatter(
            x=positions,
            y=bob_bits,
            mode='markers+lines',
            name="Bob's Bits",
            marker=dict(color='red', size=10),
            line=dict(color='red', width=2)
        ))
        
        # Highlight errors
        errors = alice_bits != bob_bits
        if np.any(errors):
            error_positions = np.array(positions)[errors]
            error_values = alice_bits[errors]
            
            fig.add_trace(go.Scatter(
                x=error_positions,
                y=error_values,
                mode='markers',
                name='Errors',
                marker=dict(color='red', size=15, symbol='x'),
                showlegend=True
            ))
        
        fig.update_layout(
            title='Sifted Key Comparison: Alice vs Bob',
            xaxis_title='Bit Position',
            yaxis_title='Bit Value',
            yaxis=dict(tickvals=[0, 1], ticktext=['0', '1']),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_protocol_diagram(self):
        """Render BB84 protocol flow diagram"""
        st.subheader("📋 Protocol Flow")
        
        st.markdown("""
        ```
        Alice                           Bob
          |                             |
          | 1. Generate bits & bases    |
          |                             |
          | 2. Prepare quantum states   |
          |                             |
          | 3. Send qubits -----------> | 4. Choose bases & measure
          |                             |
          | 5. Announce bases <-------> | 5. Announce bases
          |                             |
          | 6. Keep matching positions  | 6. Keep matching positions
          |                             |
          | 7. Test subset for errors <--> 7. Test subset for errors
          |                             |
          | 8. Error correction & PA <--> 8. Error correction & PA
          |                             |
          |     Final shared key        |
        ```
        """)
    
    def render_analysis(self):
        """Render analysis and statistics"""
        st.header("📊 BB84 Protocol Analysis")
        
        # Get experiment history
        history = [
            exp for exp in st.session_state.get('experiment_history', [])
            if exp.get('module') == 'bb84'
        ]
        
        if not history:
            st.info("📈 Run some simulations to see analysis here!")
            return
        
        # Analysis of experiments
        error_rates = [exp['error_rate'] for exp in history if 'error_rate' in exp]
        sifted_lengths = [exp['sifted_length'] for exp in history if 'sifted_length' in exp]
        
        if error_rates:
            # Error rate statistics
            avg_error_rate = np.mean(error_rates)
            st.subheader("📈 Error Rate Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Error Rate", f"{avg_error_rate:.2%}")
            
            with col2:
                st.metric("Min Error Rate", f"{min(error_rates):.2%}")
            
            with col3:
                st.metric("Max Error Rate", f"{max(error_rates):.2%}")
            
            # Error rate trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=error_rates,
                mode='lines+markers',
                name='Error Rate',
                line=dict(color='red', width=2)
            ))
            
            fig.add_hline(y=0.11, line_dash="dash", line_color="orange",
                         annotation_text="Security Threshold (11%)")
            
            fig.update_layout(
                title='Error Rate Over Experiments',
                xaxis_title='Experiment Number',
                yaxis_title='Error Rate',
                yaxis=dict(tickformat='.1%'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_advanced_features(self):
        """Render advanced BB84 features"""
        st.header("🔬 Advanced BB84 Features")
        
        feature_tabs = st.tabs([
            "Decoy States", 
            "Finite Key Analysis", 
            "Protocol Variants",
            "Implementation Details"
        ])
        
        with feature_tabs[0]:
            st.subheader("🎭 Decoy State Protocol")
            st.markdown("""
            ### Purpose
            Decoy states protect against photon-number-splitting (PNS) attacks
            in practical QKD systems using weak coherent pulses.
            
            ### Method
            - Send pulses with different intensities (signal, decoy, vacuum)
            - Monitor statistics for each intensity level
            - Detect PNS attacks through statistical analysis
            
            ### Advantages
            - Enables secure QKD with practical light sources
            - Detects sophisticated eavesdropping strategies
            - Widely used in commercial QKD systems
            """)
        
        with feature_tabs[1]:
            st.subheader("📏 Finite Key Analysis")
            st.markdown("""
            ### Challenge
            Real QKD systems generate finite-length keys, requiring statistical
            analysis of security parameters.
            
            ### Considerations
            - Statistical fluctuations in error rate estimation
            - Confidence intervals for security parameters
            - Trade-off between key length and security level
            
            ### Solutions
            - Leftover hash lemma for privacy amplification
            - Concentration inequalities for finite statistics
            - Composable security frameworks
            """)
        
        with feature_tabs[2]:
            st.subheader("🔄 Protocol Variants")
            st.markdown("""
            ### SARG04 Protocol
            - Uses only 4 quantum states (subset of BB84)
            - Better security against certain attacks
            - Lower key generation rate
            
            ### Six-State Protocol
            - Uses 6 quantum states instead of 4
            - Enhanced security against intercept-resend
            - More complex implementation
            
            ### Differential Phase Shift (DPS)
            - Uses phase encoding instead of polarization
            - Self-referencing interferometry
            - Simplified receiver design
            """)
        
        with feature_tabs[3]:
            st.subheader("⚙️ Implementation Details")
            st.markdown("""
            ### Practical Challenges
            - **Source**: Single-photon sources vs. weak coherent pulses
            - **Channel**: Fiber loss, birefringence, dispersion
            - **Detectors**: Efficiency, dark counts, dead time
            - **Synchronization**: Timing and phase stability
            
            ### Solutions
            - Decoy state methods for non-ideal sources
            - Automatic polarization compensation
            - Superconducting nanowire detectors
            - GPS synchronization and phase locking
            """)
    
    # Helper methods
    def get_state_notation(self, bit: int, basis: int) -> str:
        """Get quantum state notation for given bit and basis"""
        if basis == 0:  # Rectilinear
            return "|0⟩" if bit == 0 else "|1⟩"
        else:  # Diagonal
            return "|+⟩" if bit == 0 else "|-⟩"
    
    def create_bb84_state_circuit(self, bit: int, basis: int) -> QuantumCircuit:
        """Create quantum circuit for BB84 state preparation"""
        qc = QuantumCircuit(1, 1)
        
        if bit == 1:
            qc.x(0)  # Flip to |1⟩
        
        if basis == 1:
            qc.h(0)  # Apply Hadamard for diagonal basis
        
        qc.measure(0, 0)
        return qc

def main():
    """Main function for BB84 module"""
    module = BB84Module()
    module.main()
