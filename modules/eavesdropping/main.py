"""
Eavesdropping Detection Module
Simulates various attacks on quantum key distribution protocols
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from typing import Dict, Any, List, Tuple
from core.session_manager import SessionManager
from utils.visualizations import plot_error_analysis
from components.ui_components import ParameterControls, ResultsDisplay

class EavesdroppingModule:
    """Main eavesdropping simulation module"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.backend = AerSimulator()
        self.security_threshold = 0.11  # 11% QBER threshold for BB84
    
    def main(self):
        """Main interface for eavesdropping module"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.2em;">🕵️ Eavesdropping Detection</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1em;">Quantum Security Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main tabs
        tabs = st.tabs([
            "🎭 Attack Simulation", 
            "📊 Security Analysis", 
            "🛡️ Detection Methods",
            "📚 Attack Gallery"
        ])
        
        with tabs[0]:
            self.render_attack_simulation()
        
        with tabs[1]:
            self.render_security_analysis()
        
        with tabs[2]:
            self.render_detection_methods()
        
        with tabs[3]:
            self.render_attack_gallery()
    
    def render_attack_simulation(self):
        """Interactive attack simulation interface"""
        st.header("🎭 Quantum Attack Simulation")
        
        # Attack selection
        attack_type = st.selectbox(
            "Select Attack Strategy",
            ["Intercept-Resend", "Beam Splitter", "Photon Number Splitting", "Trojan Horse"]
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Attack Parameters")
            
            # Protocol parameters
            n_bits = st.slider("Key Length", 20, 100, 40)
            
            if attack_type == "Intercept-Resend":
                eve_measurement_prob = st.slider("Eve's Interception Rate", 0.0, 1.0, 0.5)
                eve_basis_strategy = st.selectbox("Eve's Basis Strategy", 
                                                ["Random", "Optimal", "Correlated"])
            elif attack_type == "Beam Splitter":
                splitting_ratio = st.slider("Beam Splitting Ratio", 0.1, 0.9, 0.5)
                detection_efficiency = st.slider("Detector Efficiency", 0.5, 1.0, 0.8)
            elif attack_type == "Photon Number Splitting":
                pulse_intensity = st.slider("Pulse Intensity (photons)", 0.1, 2.0, 1.0)
                splitting_efficiency = st.slider("PNS Efficiency", 0.1, 1.0, 0.6)
            
            if st.button("🚀 Launch Attack Simulation", type="primary"):
                self.simulate_attack(attack_type, n_bits, locals())
        
        with col2:
            self.render_attack_visualization(attack_type)
    
    def simulate_attack(self, attack_type: str, n_bits: int, params: Dict):
        """Simulate various eavesdropping attacks"""
        
        # Generate Alice's random bits and bases
        alice_bits = np.random.randint(0, 2, n_bits)
        alice_bases = np.random.randint(0, 2, n_bits)  # 0: rectilinear, 1: diagonal
        bob_bases = np.random.randint(0, 2, n_bits)
        
        if attack_type == "Intercept-Resend":
            results = self.intercept_resend_attack(alice_bits, alice_bases, bob_bases, params)
        elif attack_type == "Beam Splitter":
            results = self.beam_splitter_attack(alice_bits, alice_bases, bob_bases, params)
        elif attack_type == "Photon Number Splitting":
            results = self.pns_attack(alice_bits, alice_bases, bob_bases, params)
        else:
            results = self.intercept_resend_attack(alice_bits, alice_bases, bob_bases, params)
        
        self.display_attack_results(results, attack_type)
        
        # Update session
        self.session_manager.add_experiment_record('eavesdropping', {
            'attack_type': attack_type,
            'parameters': params,
            'results': results
        })
    
    def intercept_resend_attack(self, alice_bits: np.ndarray, alice_bases: np.ndarray, 
                              bob_bases: np.ndarray, params: Dict) -> Dict:
        """Simulate intercept-resend attack"""
        
        eve_prob = params.get('eve_measurement_prob', 0.5)
        n_bits = len(alice_bits)
        
        # Eve randomly chooses measurement bases
        eve_bases = np.random.randint(0, 2, n_bits)
        
        # Eve's measurements (when she intercepts)
        intercepted = np.random.random(n_bits) < eve_prob
        eve_measurements = np.zeros(n_bits)
        
        # Simulate Eve's measurement and resending
        bob_measurements = np.zeros(n_bits)
        
        for i in range(n_bits):
            if intercepted[i]:
                # Eve measures and resends
                if alice_bases[i] == eve_bases[i]:
                    # Correct basis - Eve gets correct result
                    eve_measurements[i] = alice_bits[i]
                else:
                    # Wrong basis - Eve gets random result
                    eve_measurements[i] = np.random.randint(0, 2)
                
                # Eve resends based on her measurement
                if bob_bases[i] == eve_bases[i]:
                    bob_measurements[i] = eve_measurements[i]
                else:
                    # Bob measures in different basis than Eve resent
                    bob_measurements[i] = np.random.randint(0, 2)
            else:
                # No interception - direct transmission
                if alice_bases[i] == bob_bases[i]:
                    bob_measurements[i] = alice_bits[i]
                else:
                    bob_measurements[i] = np.random.randint(0, 2)
        
        # Basis sifting
        matching_bases = alice_bases == bob_bases
        sifted_alice = alice_bits[matching_bases]
        sifted_bob = bob_measurements[matching_bases]
        
        # Calculate error rate
        if len(sifted_alice) > 0:
            errors = np.sum(sifted_alice != sifted_bob)
            error_rate = errors / len(sifted_alice)
        else:
            error_rate = 0
        
        return {
            'attack_type': 'Intercept-Resend',
            'bits_sent': n_bits,
            'bits_sifted': len(sifted_alice),
            'error_rate': error_rate,
            'eve_intercept_rate': eve_prob,
            'detected': error_rate > self.security_threshold,
            'alice_bits': sifted_alice,
            'bob_bits': sifted_bob,
            'intercepted_positions': intercepted
        }
    
    def beam_splitter_attack(self, alice_bits: np.ndarray, alice_bases: np.ndarray,
                           bob_bases: np.ndarray, params: Dict) -> Dict:
        """Simulate beam splitter attack"""
        
        splitting_ratio = params.get('splitting_ratio', 0.5)
        detection_eff = params.get('detection_efficiency', 0.8)
        
        n_bits = len(alice_bits)
        
        # Simulate beam splitting
        to_eve = np.random.random(n_bits) < splitting_ratio
        to_bob = ~to_eve
        
        # Eve's measurements on her portion
        eve_bases = np.random.randint(0, 2, n_bits)
        eve_detections = to_eve & (np.random.random(n_bits) < detection_eff)
        
        # Bob's measurements
        bob_detections = to_bob & (np.random.random(n_bits) < detection_eff)
        bob_measurements = np.zeros(n_bits)
        
        for i in range(n_bits):
            if bob_detections[i]:
                if alice_bases[i] == bob_bases[i]:
                    bob_measurements[i] = alice_bits[i]
                else:
                    bob_measurements[i] = np.random.randint(0, 2)
        
        # Calculate statistics
        detected_positions = bob_detections
        basis_matches = alice_bases == bob_bases
        valid_detections = detected_positions & basis_matches
        
        if np.sum(valid_detections) > 0:
            sifted_alice = alice_bits[valid_detections]
            sifted_bob = bob_measurements[valid_detections]
            error_rate = np.sum(sifted_alice != sifted_bob) / len(sifted_alice)
        else:
            error_rate = 0
            sifted_alice = np.array([])
            sifted_bob = np.array([])
        
        return {
            'attack_type': 'Beam Splitter',
            'bits_sent': n_bits,
            'bits_detected': np.sum(detected_positions),
            'bits_sifted': len(sifted_alice),
            'error_rate': error_rate,
            'splitting_ratio': splitting_ratio,
            'detected': False,  # Passive attack, harder to detect
            'eve_information': np.sum(eve_detections) / n_bits,
            'alice_bits': sifted_alice,
            'bob_bits': sifted_bob
        }
    
    def pns_attack(self, alice_bits: np.ndarray, alice_bases: np.ndarray,
                   bob_bases: np.ndarray, params: Dict) -> Dict:
        """Simulate Photon Number Splitting attack"""
        
        pulse_intensity = params.get('pulse_intensity', 1.0)
        pns_efficiency = params.get('splitting_efficiency', 0.6)
        
        n_bits = len(alice_bits)
        
        # Multi-photon probability (Poisson distribution)
        multi_photon_prob = 1 - np.exp(-pulse_intensity) * (1 + pulse_intensity)
        
        # PNS attack opportunities
        pns_opportunities = np.random.random(n_bits) < multi_photon_prob
        successful_pns = pns_opportunities & (np.random.random(n_bits) < pns_efficiency)
        
        # Eve stores photons for later measurement
        eve_stored = successful_pns
        bob_measurements = np.zeros(n_bits)
        
        # Bob receives remaining photons
        for i in range(n_bits):
            if not successful_pns[i]:
                # Normal transmission
                if alice_bases[i] == bob_bases[i]:
                    bob_measurements[i] = alice_bits[i]
                else:
                    bob_measurements[i] = np.random.randint(0, 2)
            else:
                # Reduced signal strength
                if alice_bases[i] == bob_bases[i]:
                    # Some probability of correct measurement despite attack
                    if np.random.random() < 0.7:  # Reduced fidelity
                        bob_measurements[i] = alice_bits[i]
                    else:
                        bob_measurements[i] = np.random.randint(0, 2)
                else:
                    bob_measurements[i] = np.random.randint(0, 2)
        
        # Basis sifting
        matching_bases = alice_bases == bob_bases
        sifted_alice = alice_bits[matching_bases]
        sifted_bob = bob_measurements[matching_bases]
        
        # Calculate error rate
        if len(sifted_alice) > 0:
            errors = np.sum(sifted_alice != sifted_bob)
            error_rate = errors / len(sifted_alice)
        else:
            error_rate = 0
        
        return {
            'attack_type': 'Photon Number Splitting',
            'bits_sent': n_bits,
            'bits_sifted': len(sifted_alice),
            'error_rate': error_rate,
            'pns_opportunities': np.sum(pns_opportunities),
            'successful_pns': np.sum(successful_pns),
            'detected': error_rate > self.security_threshold,
            'eve_information_gain': np.sum(eve_stored) / n_bits,
            'alice_bits': sifted_alice,
            'bob_bits': sifted_bob
        }
    
    def display_attack_results(self, results: Dict, attack_type: str):
        """Display attack simulation results"""
        st.subheader(f"🎯 {attack_type} Attack Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Bits Sent", results['bits_sent'])
        
        with col2:
            st.metric("Bits Sifted", results['bits_sifted'])
        
        with col3:
            st.metric("Error Rate", f"{results['error_rate']:.1%}")
        
        with col4:
            if results.get('detected', False):
                st.metric("Status", "🚨 DETECTED", delta="Attack discovered!")
            else:
                st.metric("Status", "😈 UNDETECTED", delta="Attack successful")
        
        # Security analysis
        if results['error_rate'] > self.security_threshold:
            st.error(f"⚠️ Error rate ({results['error_rate']:.1%}) exceeds security threshold ({self.security_threshold:.1%})")
            st.error("🛑 Communication should be aborted!")
        else:
            st.success(f"✅ Error rate below threshold - communication appears secure")
        
        # Attack-specific metrics
        if attack_type == "Intercept-Resend":
            st.info(f"🕵️ Eve intercepted {results['eve_intercept_rate']:.1%} of transmissions")
        elif attack_type == "Beam Splitter":
            st.info(f"📡 {results['splitting_ratio']:.1%} of photons diverted to Eve")
            st.info(f"📊 Eve gained information on {results.get('eve_information', 0):.1%} of bits")
        elif attack_type == "Photon Number Splitting":
            st.info(f"🔬 {results['pns_opportunities']} multi-photon opportunities")
            st.info(f"✅ {results['successful_pns']} successful PNS attacks")
        
        # Bit comparison visualization
        self.create_attack_visualization(results)
    
    def create_attack_visualization(self, results: Dict):
        """Create visualization of attack results"""
        
        alice_bits = results.get('alice_bits', [])
        bob_bits = results.get('bob_bits', [])
        
        if len(alice_bits) > 0:
            # Bit comparison
            fig = go.Figure()
            
            x_positions = list(range(len(alice_bits)))
            
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=alice_bits,
                mode='markers+lines',
                name="Alice's Bits",
                marker=dict(color='blue', size=8),
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=bob_bits,
                mode='markers+lines',
                name="Bob's Bits",
                marker=dict(color='red', size=8),
                line=dict(color='red', width=2)
            ))
            
            # Highlight errors
            errors = alice_bits != bob_bits
            if np.any(errors):
                error_positions = np.array(x_positions)[errors]
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
                title='Bit Comparison: Alice vs Bob',
                xaxis_title='Bit Position',
                yaxis_title='Bit Value',
                yaxis=dict(tickvals=[0, 1], ticktext=['0', '1']),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_attack_visualization(self, attack_type: str):
        """Render attack method visualization"""
        st.subheader(f"🎭 {attack_type} Attack Method")
        
        if attack_type == "Intercept-Resend":
            st.markdown("""
            ```
            Alice  ---|🔵|---  🕵️ Eve  ---|🔴|---  Bob
                     Photon      Measure     Resend
            ```
            
            **Strategy**: Eve intercepts photons, measures them, and resends
            **Detection**: Introduces ~25% error rate when undetected
            **Countermeasures**: Error rate monitoring, decoy states
            """)
        
        elif attack_type == "Beam Splitter":
            st.markdown("""
            ```
            Alice  ---|🔵|---  🪟 Beam Splitter
                                    |        \\
                                    |         \\
                               🕵️ Eve        Bob
            ```
            
            **Strategy**: Passive attack using beam splitter
            **Detection**: Difficult to detect, reduces Bob's detection rate
            **Countermeasures**: Monitoring detection statistics
            """)
        
        elif attack_type == "Photon Number Splitting":
            st.markdown("""
            ```
            Alice  ---|🔵🔵|---  🔬 PNS Device  ---|🔵|---  Bob
                     Multi-photon    Store one      Forward one
            ```
            
            **Strategy**: Exploit multi-photon pulses in practical systems
            **Detection**: Very difficult - awaits basis revelation
            **Countermeasures**: True single-photon sources, decoy states
            """)
    
    def render_security_analysis(self):
        """Security analysis tools"""
        st.header("📊 Quantum Security Analysis")
        
        # QBER monitoring
        st.subheader("📈 Error Rate Monitoring")
        
        if st.button("📊 Generate QBER Analysis"):
            self.generate_qber_analysis()
        
        # Information theory analysis
        st.subheader("🧮 Information Theoretic Security")
        
        with st.expander("Security Bound Calculator"):
            qber = st.slider("Observed QBER", 0.0, 0.25, 0.05, 0.01)
            efficiency = st.slider("Error Correction Efficiency", 1.0, 1.5, 1.16)
            
            # Calculate secure key rate
            if qber > 0 and qber < 0.5:
                h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber) if qber > 0 else 0
                secure_rate = 1 - efficiency * h_qber - h_qber
                
                st.metric("Binary Entropy H(QBER)", f"{h_qber:.3f}")
                st.metric("Secure Key Rate", f"{max(0, secure_rate):.3f}")
                
                if secure_rate > 0:
                    st.success("✅ Positive secure key rate - communication viable")
                else:
                    st.error("❌ Negative secure key rate - communication not secure")
    
    def generate_qber_analysis(self):
        """Generate QBER analysis over time"""
        # Simulate QBER measurements over time
        time_points = np.arange(1, 101)
        
        # Normal operation
        normal_qber = 0.02 + 0.01 * np.random.randn(60)
        normal_qber = np.clip(normal_qber, 0, 0.1)
        
        # Attack period
        attack_qber = 0.15 + 0.02 * np.random.randn(40)
        attack_qber = np.clip(attack_qber, 0.1, 0.3)
        
        all_qber = np.concatenate([normal_qber, attack_qber])
        
        # Create visualization
        fig = go.Figure()
        
        # Normal period
        fig.add_trace(go.Scatter(
            x=time_points[:60],
            y=normal_qber,
            mode='lines+markers',
            name='Normal Operation',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ))
        
        # Attack period
        fig.add_trace(go.Scatter(
            x=time_points[60:],
            y=attack_qber,
            mode='lines+markers',
            name='Under Attack',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        
        # Security threshold
        fig.add_hline(
            y=self.security_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Security Threshold ({self.security_threshold:.1%})"
        )
        
        fig.update_layout(
            title='Quantum Bit Error Rate Over Time',
            xaxis_title='Time (arbitrary units)',
            yaxis_title='QBER',
            yaxis=dict(tickformat='.1%'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis
        attack_detected_time = 61  # When QBER exceeded threshold
        st.warning(f"🚨 Attack detected at time point {attack_detected_time}")
        st.info("💡 Recommendation: Abort key exchange and investigate channel security")
    
    def render_detection_methods(self):
        """Detection method explanations"""
        st.header("🛡️ Eavesdropping Detection Methods")
        
        detection_tabs = st.tabs(["Statistical Tests", "Decoy States", "Entanglement", "Device Independent"])
        
        with detection_tabs[0]:
            st.subheader("📊 Statistical Detection")
            st.markdown("""
            ### Error Rate Monitoring
            - **Threshold**: QBER > 11% indicates eavesdropping
            - **Sample Size**: Need sufficient statistics for reliable detection
            - **False Positives**: Channel noise can mimic attacks
            
            ### Chi-Square Test
            - Compare observed vs expected error patterns
            - Detect non-random error distributions
            - Account for statistical fluctuations
            """)
            
            if st.button("🧪 Run Statistical Test Demo"):
                self.statistical_test_demo()
        
        with detection_tabs[1]:
            st.subheader("🎭 Decoy State Method")
            st.markdown("""
            ### Concept
            - Send photon pulses with different intensities
            - Monitor statistics for each intensity level
            - Detect photon number splitting attacks
            
            ### Implementation
            - **Signal States**: μ ≈ 0.5-1.0 photons/pulse
            - **Decoy States**: ν ≈ 0.1-0.2 photons/pulse  
            - **Vacuum States**: 0 photons/pulse
            
            ### Detection Capability
            - Can detect PNS attacks reliably
            - Enables secure QKD with weak coherent pulses
            """)
        
        with detection_tabs[2]:
            st.subheader("🔗 Entanglement-Based Detection")
            st.markdown("""
            ### Bell Inequality Tests
            - Use entangled photon pairs
            - Test for Bell inequality violations
            - Any eavesdropping reduces entanglement
            
            ### E91 Protocol
            - Based on EPR pairs instead of single photons
            - Security from Bell's theorem
            - Inherent eavesdropping detection
            """)
        
        with detection_tabs[3]:
            st.subheader("🔒 Device-Independent QKD")
            st.markdown("""
            ### Motivation
            - No assumptions about device implementations
            - Security even with imperfect devices
            - Protection against device-specific attacks
            
            ### CHSH Inequality
            - S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| ≤ 2
            - Quantum mechanics allows S ≤ 2√2 ≈ 2.83
            - Violation proves quantum behavior
            """)
    
    def statistical_test_demo(self):
        """Demonstrate statistical test for eavesdropping"""
        st.subheader("📈 Statistical Test Results")
        
        # Generate test data
        n_samples = 1000
        normal_errors = np.random.binomial(1, 0.05, n_samples)  # 5% natural error rate
        attack_errors = np.random.binomial(1, 0.15, n_samples)  # 15% with attack
        
        # Chi-square test simulation
        observed_normal = np.sum(normal_errors)
        observed_attack = np.sum(attack_errors)
        expected = n_samples * 0.05  # Expected under null hypothesis
        
        chi2_normal = (observed_normal - expected)**2 / expected
        chi2_attack = (observed_attack - expected)**2 / expected
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Normal Channel")
            st.metric("Observed Errors", observed_normal)
            st.metric("Chi-Square Statistic", f"{chi2_normal:.2f}")
            if chi2_normal < 3.84:  # 95% confidence threshold
                st.success("✅ No eavesdropping detected")
            else:
                st.warning("⚠️ Possible eavesdropping")
        
        with col2:
            st.subheader("Under Attack")
            st.metric("Observed Errors", observed_attack)
            st.metric("Chi-Square Statistic", f"{chi2_attack:.2f}")
            if chi2_attack < 3.84:
                st.success("✅ No eavesdropping detected")
            else:
                st.error("🚨 Eavesdropping detected!")
    
    def render_attack_gallery(self):
        """Gallery of different attack methods"""
        st.header("📚 Attack Method Gallery")
        
        attack_gallery = {
            "Individual Attacks": {
                "Intercept-Resend": "Classic attack with 25% error signature",
                "Beam Splitter": "Passive attack using optical components", 
                "Photon Number Splitting": "Exploits multi-photon pulses",
                "Trojan Horse": "Backflow attack using bright light"
            },
            "Coherent Attacks": {
                "Optimal Cloning": "Quantum cloning with optimal fidelity",
                "Collective Measurements": "Joint measurements on multiple qubits",
                "Entangling Cloner": "Uses entanglement for information extraction"
            },
            "Implementation Attacks": {
                "Detector Blinding": "Exploit detector vulnerabilities",
                "Time-Shift Attack": "Exploit timing in practical devices",
                "Wavelength Attack": "Exploit wavelength dependencies"
            }
        }
        
        for category, attacks in attack_gallery.items():
            st.subheader(f"🎭 {category}")
            
            for attack_name, description in attacks.items():
                with st.expander(f"{attack_name}"):
                    st.write(description)
                    
                    if attack_name in ["Intercept-Resend", "Beam Splitter", "Photon Number Splitting"]:
                        if st.button(f"Try {attack_name}", key=f"gallery_{attack_name}"):
                            st.info(f"Switch to Attack Simulation tab to try {attack_name}")

def main():
    """Main function for eavesdropping module"""
    module = EavesdroppingModule()
    module.main()
