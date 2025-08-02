"""
Security Analysis Module for Quantum Cryptography
Advanced security analysis and optimization tools
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple
from core.session_manager import SessionManager
import scipy.optimize as opt
from scipy.special import erfc

class SecurityAnalysisModule:
    """Main security analysis module"""
    
    def __init__(self):
        self.session_manager = SessionManager()
    
    def main(self):
        """Main interface for security analysis module"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.2em;">🛡️ Security Analysis</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1em;">Quantum Cryptographic Security Assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main tabs
        tabs = st.tabs([
            "🔐 Key Rate Analysis", 
            "📊 Security Bounds", 
            "🎯 Parameter Optimization",
            "🔬 Advanced Analysis"
        ])
        
        with tabs[0]:
            self.render_key_rate_analysis()
        
        with tabs[1]:
            self.render_security_bounds()
        
        with tabs[2]:
            self.render_parameter_optimization()
        
        with tabs[3]:
            self.render_advanced_analysis()
    
    def render_key_rate_analysis(self):
        """Key rate analysis interface"""
        st.header("🔐 Secure Key Rate Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Protocol Parameters")
            
            # Basic parameters
            protocol = st.selectbox("Protocol", ["BB84", "SARG04", "Six-State"])
            
            # Channel parameters
            st.subheader("📡 Channel Properties")
            distance = st.slider("Distance (km)", 0, 200, 50)
            loss_coefficient = st.slider("Loss Coefficient (dB/km)", 0.1, 1.0, 0.2)
            
            # System parameters
            st.subheader("⚙️ System Parameters")
            detector_efficiency = st.slider("Detector Efficiency", 0.1, 1.0, 0.5)
            dark_count_rate = st.slider("Dark Count Rate (Hz)", 1e-6, 1e-3, 1e-5, format="%.2e")
            error_correction_efficiency = st.slider("EC Efficiency", 1.0, 2.0, 1.16)
            
            if st.button("📈 Calculate Key Rate", type="primary"):
                self.calculate_key_rate(protocol, distance, loss_coefficient, 
                                      detector_efficiency, dark_count_rate, 
                                      error_correction_efficiency)
        
        with col2:
            self.render_key_rate_visualization()
    
    def calculate_key_rate(self, protocol: str, distance: float, loss_coeff: float,
                          det_eff: float, dark_rate: float, ec_eff: float):
        """Calculate secure key rate"""
        
        # Calculate channel transmittance
        transmittance = 10 ** (-loss_coeff * distance / 10)
        
        # Detection probability
        p_det = det_eff * transmittance
        
        # Error rate calculation (simplified model)
        qber = self.calculate_qber(p_det, dark_rate, protocol)
        
        # Binary entropy function
        def h_binary(x):
            if x <= 0 or x >= 1:
                return 0
            return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
        
        # Secure key rate (Devetak-Winter bound)
        if qber < 0.5:
            h_qber = h_binary(qber)
            
            # Protocol-specific efficiency
            if protocol == "BB84":
                sifting_efficiency = 0.5  # Half the bits are sifted
            elif protocol == "SARG04":
                sifting_efficiency = 0.25  # Quarter of bits are sifted
            else:  # Six-State
                sifting_efficiency = 1/3   # One third of bits are sifted
            
            # Key rate calculation
            raw_rate = sifting_efficiency * p_det  # Simplified
            secure_rate = raw_rate * (1 - ec_eff * h_qber - h_qber)
            secure_rate = max(0, secure_rate)  # Can't be negative
        else:
            secure_rate = 0
            h_qber = 0.5
            raw_rate = 0
        
        # Display results
        self.display_key_rate_results({
            'protocol': protocol,
            'distance': distance,
            'transmittance': transmittance,
            'qber': qber,
            'binary_entropy': h_qber,
            'raw_rate': raw_rate,
            'secure_rate': secure_rate,
            'viable': secure_rate > 0
        })
        
        # Update session
        self.session_manager.add_experiment_record('security_analysis', {
            'analysis_type': 'key_rate',
            'protocol': protocol,
            'parameters': {
                'distance': distance,
                'loss_coefficient': loss_coeff,
                'detector_efficiency': det_eff,
                'dark_count_rate': dark_rate,
                'ec_efficiency': ec_eff
            },
            'results': {
                'qber': qber,
                'secure_rate': secure_rate
            }
        })
    
    def calculate_qber(self, p_det: float, dark_rate: float, protocol: str) -> float:
        """Calculate quantum bit error rate"""
        
        # Simplified QBER model
        # Intrinsic error rate (polarization drift, etc.)
        intrinsic_error = 0.01
        
        # Dark count contribution
        if p_det > 0:
            dark_error_contribution = dark_rate / (p_det + dark_rate) * 0.5
        else:
            dark_error_contribution = 0.5
        
        # Protocol-specific error rates
        if protocol == "BB84":
            protocol_factor = 1.0
        elif protocol == "SARG04":
            protocol_factor = 0.5  # Lower error rate
        else:  # Six-State
            protocol_factor = 1.2  # Slightly higher error rate
        
        # Total QBER calculation
        total_qber = intrinsic_error * protocol_factor + dark_error_contribution
        
        # Ensure QBER is within physical bounds
        return min(max(total_qber, 0.0), 0.5)
    
    def display_key_rate_results(self, results: Dict):
        """Display key rate calculation results"""
        st.subheader("🎯 Key Rate Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Distance", f"{results['distance']} km")
            st.metric("Transmittance", f"{results['transmittance']:.6f}")
        
        with col2:
            st.metric("QBER", f"{results['qber']:.3%}")
            st.metric("Binary Entropy", f"{results['binary_entropy']:.3f}")
        
        with col3:
            st.metric("Raw Key Rate", f"{results['raw_rate']:.6f}")
            st.metric("Secure Key Rate", f"{results['secure_rate']:.6f}")
        
        with col4:
            if results['viable']:
                st.success("✅ VIABLE")
                st.metric("Status", "Secure", delta="Communication possible")
            else:
                st.error("❌ NOT VIABLE")
                st.metric("Status", "Insecure", delta="No secure key possible")
        
        # Security analysis
        if results['qber'] > 0.11:
            st.warning(f"⚠️ QBER ({results['qber']:.1%}) exceeds typical security threshold (11%)")
        
        if results['secure_rate'] > 0:
            st.info(f"💡 At this distance, you can generate {results['secure_rate']:.6f} secure bits per transmitted photon")
        else:
            st.error("🚫 No secure communication possible at this distance with current parameters")
    
    def render_key_rate_visualization(self):
        """Render key rate vs distance visualization"""
        st.subheader("📈 Key Rate vs Distance")
        
        # Generate data for visualization
        distances = np.linspace(0, 200, 50)
        secure_rates = []
        qbers = []
        
        for dist in distances:
            transmittance = 10 ** (-0.2 * dist / 10)  # Default loss coefficient
            p_det = 0.5 * transmittance  # Default detector efficiency
            qber = self.calculate_qber(p_det, 1e-5, "BB84")
            
            # Calculate secure rate
            if qber < 0.5:
                h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber) if qber > 0 else 0
                raw_rate = 0.5 * p_det
                secure_rate = raw_rate * (1 - 1.16 * h_qber - h_qber)
                secure_rate = max(0, secure_rate)
            else:
                secure_rate = 0
            
            secure_rates.append(secure_rate)
            qbers.append(qber)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Secure Key Rate vs Distance', 'QBER vs Distance'),
            vertical_spacing=0.15
        )
        
        # Secure key rate plot
        fig.add_trace(go.Scatter(
            x=distances,
            y=secure_rates,
            mode='lines',
            name='Secure Key Rate',
            line=dict(color='green', width=3)
        ), row=1, col=1)
        
        # QBER plot
        fig.add_trace(go.Scatter(
            x=distances,
            y=qbers,
            mode='lines',
            name='QBER',
            line=dict(color='red', width=3)
        ), row=2, col=1)
        
        # Add security threshold line
        fig.add_hline(y=0.11, line_dash="dash", line_color="orange", 
                     annotation_text="Security Threshold (11%)", row=2, col=1)
        
        fig.update_layout(height=500, title_text="QKD Performance Analysis")
        fig.update_xaxes(title_text="Distance (km)")
        fig.update_yaxes(title_text="Secure Key Rate", row=1, col=1)
        fig.update_yaxes(title_text="QBER", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_security_bounds(self):
        """Security bounds analysis"""
        st.header("📊 Information-Theoretic Security Bounds")
        
        # Security bound calculator
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧮 Security Bound Calculator")
            
            qber_input = st.slider("QBER", 0.0, 0.25, 0.05, 0.01)
            ec_efficiency = st.slider("Error Correction Efficiency", 1.0, 2.0, 1.16)
            security_parameter = st.number_input("Security Parameter (ε)", 
                                               min_value=1e-15, max_value=1e-6, 
                                               value=1e-9, format="%.2e")
            
            if st.button("🔒 Calculate Security Bounds"):
                self.calculate_security_bounds(qber_input, ec_efficiency, security_parameter)
        
        with col2:
            self.render_security_bounds_visualization()
    
    def calculate_security_bounds(self, qber: float, ec_eff: float, eps: float):
        """Calculate information-theoretic security bounds"""
        
        # Binary entropy function
        def h_binary(x):
            if x <= 0 or x >= 1:
                return 0
            return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
        
        # Asymptotic key rate
        h_qber = h_binary(qber)
        asymptotic_rate = 1 - ec_eff * h_qber - h_qber
        
        # Finite key corrections (simplified)
        finite_key_correction = np.sqrt(-np.log2(eps) / 1000)  # Assuming 1000 bits
        finite_rate = asymptotic_rate - finite_key_correction
        
        # Display results
        st.subheader("🔒 Security Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Binary Entropy H(QBER)", f"{h_qber:.4f}")
            st.metric("Asymptotic Rate", f"{asymptotic_rate:.4f}")
        
        with col2:
            st.metric("Finite Key Correction", f"{finite_key_correction:.4f}")
            st.metric("Finite Key Rate", f"{max(0, finite_rate):.4f}")
        
        # Security assessment
        if asymptotic_rate > 0:
            st.success("✅ Asymptotically secure communication possible")
        else:
            st.error("❌ No asymptotic security achievable")
        
        if finite_rate > 0:
            st.success("✅ Finite-key security achievable")
        else:
            st.warning("⚠️ Finite-key effects prevent secure communication")
    
    def render_security_bounds_visualization(self):
        """Visualize security bounds"""
        st.subheader("📈 Security Bounds Visualization")
        
        # Generate QBER range
        qbers = np.linspace(0, 0.25, 100)
        
        # Calculate bounds for different protocols
        bb84_rates = []
        sarg04_rates = []
        
        for qber in qbers:
            h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber) if qber > 0 else 0
            
            # BB84 rate
            bb84_rate = 1 - 1.16 * h_qber - h_qber
            bb84_rates.append(max(0, bb84_rate))
            
            # SARG04 rate (different sifting efficiency)
            sarg04_rate = 0.5 * (1 - 1.16 * h_qber - h_qber)
            sarg04_rates.append(max(0, sarg04_rate))
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=qbers,
            y=bb84_rates,
            mode='lines',
            name='BB84',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=qbers,
            y=sarg04_rates,
            mode='lines',
            name='SARG04',
            line=dict(color='green', width=3)
        ))
        
        # Add threshold line
        fig.add_vline(x=0.11, line_dash="dash", line_color="red",
                     annotation_text="Typical Security Threshold")
        
        fig.update_layout(
            title='Secure Key Rate vs QBER for Different Protocols',
            xaxis_title='QBER',
            yaxis_title='Secure Key Rate',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_parameter_optimization(self):
        """Parameter optimization interface"""
        st.header("🎯 System Parameter Optimization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎛️ Optimization Targets")
            
            target_distance = st.slider("Target Distance (km)", 10, 200, 100)
            optimization_goal = st.selectbox("Optimization Goal", 
                                           ["Maximize Key Rate", "Minimize QBER", "Maximize Range"])
            
            # Constraints
            st.subheader("📋 Constraints")
            max_detector_cost = st.slider("Max Detector Efficiency", 0.5, 1.0, 0.9)
            max_ec_overhead = st.slider("Max EC Overhead", 1.0, 2.0, 1.5)
            
            if st.button("🚀 Run Optimization", type="primary"):
                self.run_parameter_optimization(target_distance, optimization_goal, 
                                              max_detector_cost, max_ec_overhead)
        
        with col2:
            self.render_optimization_visualization()
    
    def run_parameter_optimization(self, distance: float, goal: str, 
                                 max_det_eff: float, max_ec_eff: float):
        """Run parameter optimization"""
        
        def objective(params):
            det_eff, ec_eff = params
            
            # Calculate performance metrics
            transmittance = 10 ** (-0.2 * distance / 10)
            p_det = det_eff * transmittance
            qber = self.calculate_qber(p_det, 1e-5, "BB84")
            
            if qber < 0.5:
                h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber) if qber > 0 else 0
                raw_rate = 0.5 * p_det
                secure_rate = raw_rate * (1 - ec_eff * h_qber - h_qber)
                secure_rate = max(0, secure_rate)
            else:
                secure_rate = 0
            
            # Optimization objectives
            if goal == "Maximize Key Rate":
                return -secure_rate  # Minimize negative rate
            elif goal == "Minimize QBER":
                return qber
            else:  # Maximize Range
                return -secure_rate if secure_rate > 0.001 else 1000  # Penalty for very low rates
        
        # Optimization bounds
        bounds = [(0.1, max_det_eff), (1.0, max_ec_eff)]
        
        # Run optimization
        result = opt.minimize(objective, x0=[0.5, 1.16], bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            opt_det_eff, opt_ec_eff = result.x
            
            # Calculate final metrics
            transmittance = 10 ** (-0.2 * distance / 10)
            p_det = opt_det_eff * transmittance
            qber = self.calculate_qber(p_det, 1e-5, "BB84")
            
            if qber < 0.5:
                h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber) if qber > 0 else 0
                raw_rate = 0.5 * p_det
                secure_rate = raw_rate * (1 - opt_ec_eff * h_qber - h_qber)
                secure_rate = max(0, secure_rate)
            else:
                secure_rate = 0
            
            # Display results
            st.subheader("🎯 Optimization Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Optimal Detector Efficiency", f"{opt_det_eff:.3f}")
                st.metric("Optimal EC Efficiency", f"{opt_ec_eff:.3f}")
            
            with col2:
                st.metric("Resulting QBER", f"{qber:.3%}")
                st.metric("Detection Probability", f"{p_det:.6f}")
            
            with col3:
                st.metric("Secure Key Rate", f"{secure_rate:.6f}")
                if secure_rate > 0:
                    st.success("✅ Optimization successful")
                else:
                    st.error("❌ No secure communication possible")
        else:
            st.error("❌ Optimization failed to converge")
    
    def render_optimization_visualization(self):
        """Render optimization parameter space"""
        st.subheader("🗺️ Parameter Space Exploration")
        
        # Create parameter grid
        det_effs = np.linspace(0.1, 1.0, 20)
        ec_effs = np.linspace(1.0, 2.0, 20)
        
        # Calculate key rates for parameter grid
        key_rates = np.zeros((len(det_effs), len(ec_effs)))
        
        for i, det_eff in enumerate(det_effs):
            for j, ec_eff in enumerate(ec_effs):
                transmittance = 10 ** (-0.2 * 100 / 10)  # 100 km distance
                p_det = det_eff * transmittance
                qber = self.calculate_qber(p_det, 1e-5, "BB84")
                
                if qber < 0.5:
                    h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber) if qber > 0 else 0
                    raw_rate = 0.5 * p_det
                    secure_rate = raw_rate * (1 - ec_eff * h_qber - h_qber)
                    key_rates[i, j] = max(0, secure_rate)
                else:
                    key_rates[i, j] = 0
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=key_rates,
            x=ec_effs,
            y=det_effs,
            colorscale='Viridis',
            colorbar=dict(title="Secure Key Rate")
        ))
        
        fig.update_layout(
            title='Secure Key Rate vs System Parameters (100 km)',
            xaxis_title='Error Correction Efficiency',
            yaxis_title='Detector Efficiency',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_advanced_analysis(self):
        """Advanced security analysis tools"""
        st.header("🔬 Advanced Security Analysis")
        
        analysis_tabs = st.tabs(["Finite Key Effects", "Composable Security", "Side Channels", "Future Protocols"])
        
        with analysis_tabs[0]:
            self.render_finite_key_analysis()
        
        with analysis_tabs[1]:
            self.render_composable_security()
        
        with analysis_tabs[2]:
            self.render_side_channel_analysis()
        
        with analysis_tabs[3]:
            self.render_future_protocols()
    
    def render_finite_key_analysis(self):
        """Finite key size effects analysis"""
        st.subheader("📏 Finite Key Size Effects")
        
        st.markdown("""
        ### Impact of Finite Key Length
        
        Real QKD systems use finite key lengths, which affects security:
        
        - **Statistical Fluctuations**: Finite sampling leads to uncertainty in error rates
        - **Security Parameter**: ε-security requires accounting for failure probability
        - **Key Rate Reduction**: Finite-key corrections reduce achievable rates
        """)
        
        # Finite key calculator
        key_length = st.slider("Key Length", 100, 100000, 10000, step=100)
        target_security = st.selectbox("Security Level", ["10⁻⁶", "10⁻⁹", "10⁻¹²"])
        
        if st.button("🧮 Calculate Finite Key Effects"):
            eps = float(target_security.replace("⁻", "e-"))
            
            # Simplified finite key correction
            correction = np.sqrt(-np.log2(eps) / key_length)
            
            st.write(f"**Security Parameter:** ε = {eps}")
            st.write(f"**Finite Key Correction:** {correction:.6f}")
            st.write(f"**Rate Reduction:** {correction * 100:.2f}%")
            
            # Visualization of finite key effects
            key_lengths = np.logspace(2, 5, 100)  # 100 to 100,000 bits
            corrections = np.sqrt(-np.log2(eps) / key_lengths)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=key_lengths,
                y=corrections,
                mode='lines',
                name=f'ε = {eps}',
                line=dict(width=3)
            ))
            
            fig.update_layout(
                title='Finite Key Correction vs Key Length',
                xaxis_title='Key Length (bits)',
                yaxis_title='Correction Term',
                xaxis_type='log',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key length recommendations
            st.subheader("📋 Key Length Recommendations")
            
            target_corrections = [0.01, 0.005, 0.001]  # 1%, 0.5%, 0.1%
            
            recommendations = []
            for target_corr in target_corrections:
                min_length = int(-np.log2(eps) / (target_corr ** 2))
                recommendations.append({
                    'Target Correction': f"{target_corr * 100:.1f}%",
                    'Min Key Length': f"{min_length:,} bits",
                    'Recommended': f"{int(min_length * 1.5):,} bits"
                })
            
            import pandas as pd
            df = pd.DataFrame(recommendations)
            st.dataframe(df, use_container_width=True)
    
    def render_composable_security(self):
        """Composable security analysis"""
        st.subheader("🔒 Composable Security Framework")
        
        st.markdown("""
        ### Universal Composability (UC) Framework
        
        Composable security ensures that a QKD protocol remains secure when used as a 
        component in larger cryptographic systems.
        
        **Key Principles:**
        - **Simulation-based security**: Real protocol indistinguishable from ideal functionality
        - **Universal composition**: Security preserved under arbitrary composition
        - **Adaptive adversaries**: Security against adaptive attacks
        
        ### Security Parameters
        
        A QKD protocol is **(ε_c, ε_s)-secure** if:
        - **Completeness**: P(abort) ≤ ε_c (protocol completion probability)
        - **Soundness**: Output key is ε_s-close to uniform random string
        """)
        
        # Interactive composable security calculator
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧮 Composability Analysis")
            
            completeness_eps = st.number_input("Completeness ε_c", 
                                             min_value=1e-10, max_value=1e-2, 
                                             value=1e-6, format="%.2e")
            soundness_eps = st.number_input("Soundness ε_s", 
                                          min_value=1e-15, max_value=1e-6, 
                                          value=1e-9, format="%.2e")
            
            if st.button("📊 Analyze Composability"):
                # Simplified composability analysis
                total_security = completeness_eps + soundness_eps
                
                st.metric("Total Security Parameter", f"{total_security:.2e}")
                
                if total_security < 1e-8:
                    st.success("✅ Excellent composable security")
                elif total_security < 1e-6:
                    st.success("✅ Good composable security")
                elif total_security < 1e-4:
                    st.warning("⚠️ Moderate composable security")
                else:
                    st.error("❌ Poor composable security")
        
        with col2:
            st.subheader("🔗 Composition Scenarios")
            
            composition_types = [
                "Sequential Composition",
                "Parallel Composition", 
                "Concurrent Composition",
                "Self Composition"
            ]
            
            selected_composition = st.selectbox("Composition Type", composition_types)
            
            if selected_composition == "Sequential Composition":
                st.info("🔄 Keys used one after another - security parameters add linearly")
            elif selected_composition == "Parallel Composition":
                st.info("🔀 Multiple protocols run simultaneously - requires careful analysis")
            elif selected_composition == "Concurrent Composition":
                st.info("⚡ Protocols share resources - most challenging scenario")
            else:
                st.info("🔁 Same protocol run multiple times - security degrades with repetition")
    
    def render_side_channel_analysis(self):
        """Side channel attack analysis"""
        st.subheader("🕳️ Side Channel Vulnerabilities")
        
        st.markdown("""
        ### Major Side Channel Attack Categories
        
        Side channel attacks exploit implementation imperfections rather than 
        theoretical weaknesses in quantum cryptography protocols.
        """)
        
        attack_tabs = st.tabs(["Detector Attacks", "Source Attacks", "Channel Attacks", "Countermeasures"])
        
        with attack_tabs[0]:
            st.subheader("🔍 Detector Side Channels")
            
            detector_attacks = {
                "Detector Blinding": {
                    "description": "Bright light disables single-photon detectors",
                    "impact": "Eve can control measurement outcomes",
                    "detection_difficulty": "Medium",
                    "countermeasures": ["Monitor detector currents", "Use monitoring photodiodes"]
                },
                "Time-Shift Attack": {
                    "description": "Exploit timing dependencies in detector responses",
                    "impact": "Information leakage through timing correlations",
                    "detection_difficulty": "Hard",
                    "countermeasures": ["Randomize timing", "Characterize detector response"]
                },
                "Efficiency Mismatch": {
                    "description": "Different detection efficiencies for different states",
                    "impact": "Biased measurement statistics",
                    "detection_difficulty": "Easy",
                    "countermeasures": ["Detector characterization", "Efficiency correction"]
                }
            }
            
            selected_attack = st.selectbox("Select Detector Attack", list(detector_attacks.keys()))
            
            attack_info = detector_attacks[selected_attack]
            
            st.write(f"**Description:** {attack_info['description']}")
            st.write(f"**Impact:** {attack_info['impact']}")
            st.write(f"**Detection Difficulty:** {attack_info['detection_difficulty']}")
            
            st.write("**Countermeasures:**")
            for countermeasure in attack_info['countermeasures']:
                st.write(f"• {countermeasure}")
        
        with attack_tabs[1]:
            st.subheader("💡 Source Side Channels")
            
            st.markdown("""
            **Trojan Horse Attacks:**
            - Eve sends bright light back into Alice's source
            - Information leaked through backscattered light
            - Countermeasure: Optical isolators and monitoring
            
            **Wavelength Dependencies:**
            - Components have wavelength-dependent responses
            - Eve can exploit by using different wavelengths
            - Countermeasure: Wavelength filtering and monitoring
            
            **Intensity Modulation:**
            - Unintended intensity correlations with bit values
            - Eve can perform photon-number-splitting attacks
            - Countermeasure: Decoy state protocols
            """)
        
        with attack_tabs[2]:
            st.subheader("📡 Channel Side Channels")
            
            st.markdown("""
            **Channel Loss Correlations:**
            - Loss may correlate with transmitted bit values
            - Eve can gain information from loss patterns
            - Countermeasure: Loss monitoring and randomization
            
            **Polarization Drift:**
            - Fiber birefringence causes polarization rotation
            - Time-dependent effects can leak information
            - Countermeasure: Active polarization compensation
            """)
        
        with attack_tabs[3]:
            st.subheader("🛡️ Countermeasures")
            
            countermeasure_categories = {
                "Device-Independent QKD": {
                    "security_level": "Highest",
                    "implementation_difficulty": "Very Hard",
                    "description": "Security based only on Bell inequality violations"
                },
                "Measurement-Device-Independent QKD": {
                    "security_level": "High", 
                    "implementation_difficulty": "Medium",
                    "description": "Removes all detector side channels"
                },
                "Hardware Improvements": {
                    "security_level": "Medium",
                    "implementation_difficulty": "Easy",
                    "description": "Better components and monitoring"
                },
                "Protocol Modifications": {
                    "security_level": "Medium",
                    "implementation_difficulty": "Easy", 
                    "description": "Decoy states, random sampling, etc."
                }
            }
            
            for method, info in countermeasure_categories.items():
                with st.expander(f"{method}"):
                    st.write(f"**Security Level:** {info['security_level']}")
                    st.write(f"**Implementation Difficulty:** {info['implementation_difficulty']}")
                    st.write(f"**Description:** {info['description']}")
    
    def render_future_protocols(self):
        """Future QKD protocols and technologies"""
        st.subheader("🚀 Next-Generation QKD Protocols")
        
        # Protocol comparison matrix
        protocols_data = {
            "Protocol": ["BB84", "SARG04", "MDI-QKD", "Twin-Field QKD", "DI-QKD"],
            "Security Model": ["Trusted Devices", "Trusted Devices", "Untrusted Detectors", "Untrusted Relay", "No Device Trust"],
            "Key Rate": ["High", "Medium", "Medium", "High", "Low"],
            "Max Distance": ["~100 km", "~100 km", "~200 km", "~500 km", "~50 km"],
            "Implementation": ["Commercial", "Research", "Demonstrated", "Research", "Proof-of-concept"],
            "Main Advantage": ["Simplicity", "4-State Security", "Detector Security", "Long Distance", "Ultimate Security"]
        }
        
        import pandas as pd
        df = pd.DataFrame(protocols_data)
        st.dataframe(df, use_container_width=True)
        
        # Detailed protocol descriptions
        protocol_tabs = st.tabs(["MDI-QKD", "Twin-Field QKD", "Device-Independent", "Quantum Networks"])
        
        with protocol_tabs[0]:
            st.subheader("🔬 Measurement-Device-Independent QKD")
            
            st.markdown("""
            ### Key Innovation
            - **Untrusted measurement devices**: No assumptions about detectors
            - **Bell state measurements**: Performed at untrusted relay station
            - **Side channel immunity**: Removes all detector-based attacks
            
            ### Protocol Overview
            1. Alice and Bob prepare quantum states independently
            2. States sent to untrusted relay (Charlie)
            3. Charlie performs Bell state measurement
            4. Results announced publicly
            5. Alice and Bob perform error correction and privacy amplification
            
            ### Advantages
            - Immune to all detector side channels
            - Extends secure transmission distance
            - Practical with current technology
            
            ### Challenges
            - Lower key rates than direct transmission
            - Requires phase stabilization
            - More complex setup
            """)
            
            if st.button("📊 MDI-QKD Performance Analysis"):
                # Simulate MDI-QKD performance
                distances = np.linspace(0, 300, 50)
                mdi_rates = []
                
                for dist in distances:
                    # Simplified MDI-QKD rate calculation
                    transmittance = (10 ** (-0.2 * dist / 20))**2  # Two-way loss
                    detection_rate = 0.1 * transmittance  # Bell state measurement efficiency
                    
                    if detection_rate > 1e-8:
                        qber = 0.01 + 0.5 * (1 - transmittance)  # Simplified QBER model
                        if qber < 0.11:
                            h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber) if qber > 0 else 0
                            rate = detection_rate * (1 - 1.2 * h_qber - h_qber)
                            mdi_rates.append(max(0, rate))
                        else:
                            mdi_rates.append(0)
                    else:
                        mdi_rates.append(0)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=distances,
                    y=mdi_rates,
                    mode='lines',
                    name='MDI-QKD Rate',
                    line=dict(color='purple', width=3)
                ))
                
                fig.update_layout(
                    title='MDI-QKD Key Rate vs Distance',
                    xaxis_title='Distance (km)',
                    yaxis_title='Secure Key Rate',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with protocol_tabs[1]:
            st.subheader("🔗 Twin-Field QKD")
            
            st.markdown("""
            ### Revolutionary Approach
            - **Single-photon interference**: Uses interference of single photons
            - **Virtual Z-basis**: Measurement basis determined by phase relationships
            - **Rate scaling**: Better than direct transmission at long distances
            
            ### Key Features
            - Phase-randomized coherent states
            - Single-photon interference at relay
            - Post-selection based on successful interference
            - Superior distance scaling: R ∝ η instead of R ∝ η²
            
            ### Record Achievements
            - Demonstrated over 500+ km distances
            - Higher rates than repeater-based approaches
            - Compatible with existing telecom infrastructure
            """)
        
        with protocol_tabs[2]:
            st.subheader("🔒 Device-Independent QKD")
            
            st.markdown("""
            ### Ultimate Security Goal
            - **No device assumptions**: Security independent of implementation
            - **Bell test based**: Uses violation of Bell inequalities
            - **Composable security**: Rigorous security proofs
            
            ### Technical Requirements
            - High-quality entanglement sources
            - Efficient detectors with high fidelity
            - Loophole-free Bell tests
            - Advanced error correction
            
            ### Current Status
            - Proof-of-principle demonstrations
            - Very low key rates
            - Short distances (~km range)
            - Active research area
            
            ### Future Prospects
            - Improved photon sources
            - Better detectors
            - Optimized protocols
            - Network implementations
            """)
        
        with protocol_tabs[3]:
            st.subheader("🌐 Quantum Networks")
            
            st.markdown("""
            ### Network QKD Challenges
            - **Routing**: Secure routing in quantum networks
            - **Key management**: Distribution and storage of quantum keys
            - **Scalability**: Network grows with N² key pairs
            - **Trust models**: Different trust assumptions for different nodes
            
            ### Solutions in Development
            - **Quantum key servers**: Centralized key distribution
            - **Network coding**: Efficient key routing protocols
            - **Hybrid networks**: Classical-quantum integration
            - **Satellite QKD**: Global quantum communication
            
            ### Commercial Deployments
            - Metropolitan quantum networks (Vienna, Beijing, Tokyo)
            - Bank-to-bank secure communications
            - Government secure communications
            - Integration with classical networks
            """)

def main():
    """Main function for security analysis module"""
    module = SecurityAnalysisModule()
    module.main()
