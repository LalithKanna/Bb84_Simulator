"""
Error Correction Module for Quantum Cryptography
Implements classical error correction for quantum key distribution
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple
from core.session_manager import SessionManager

class ErrorCorrectionModule:
    """Main error correction module"""
    
    def __init__(self):
        self.session_manager = SessionManager()
    
    def main(self):
        """Main interface for error correction module"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.2em;">🔧 Error Correction</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1em;">Classical Error Correction for QKD</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main tabs
        tabs = st.tabs([
            "🛠️ Interactive Correction", 
            "📊 Algorithm Comparison", 
            "🔄 CASCADE Protocol",
            "📚 Theory & Practice"
        ])
        
        with tabs[0]:
            self.render_interactive_correction()
        
        with tabs[1]:
            self.render_algorithm_comparison()
        
        with tabs[2]:
            self.render_cascade_protocol()
        
        with tabs[3]:
            self.render_theory_practice()
    
    def render_interactive_correction(self):
        """Interactive error correction interface"""
        st.header("🛠️ Interactive Error Correction Laboratory")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Setup Parameters")
            
            # Key generation
            key_length = st.slider("Key Length", 20, 100, 50)
            error_rate = st.slider("Error Rate", 0.0, 0.25, 0.1, 0.01)
            
            # Error correction method
            correction_method = st.selectbox(
                "Correction Method",
                ["Hamming Code", "CASCADE", "Parity Check", "BCH Code"]
            )
            
            if st.button("🎲 Generate Test Keys", type="primary"):
                self.generate_test_scenario(key_length, error_rate, correction_method)
        
        with col2:
            self.render_correction_visualization()
    
    def generate_test_scenario(self, key_length: int, error_rate: float, method: str):
        """Generate test scenario for error correction"""
        
        # Generate Alice's key
        alice_key = np.random.randint(0, 2, key_length)
        
        # Generate Bob's key with errors
        bob_key = alice_key.copy()
        error_positions = np.random.random(key_length) < error_rate
        bob_key[error_positions] = 1 - bob_key[error_positions]  # Flip bits
        
        st.subheader("🔑 Generated Keys")
        
        # Display keys
        self.display_keys(alice_key, bob_key, error_positions)
        
        # Apply error correction
        if method == "Hamming Code":
            results = self.hamming_correction(alice_key, bob_key)
        elif method == "CASCADE":
            results = self.cascade_correction(alice_key, bob_key)
        elif method == "Parity Check":
            results = self.parity_correction(alice_key, bob_key)
        else:
            results = self.basic_correction(alice_key, bob_key)
        
        self.display_correction_results(results, method)
        
        # Update session
        self.session_manager.add_experiment_record('error_correction', {
            'method': method,
            'key_length': key_length,
            'initial_error_rate': error_rate,
            'results': results
        })
    
    def display_keys(self, alice_key: np.ndarray, bob_key: np.ndarray, errors: np.ndarray):
        """Display Alice and Bob's keys with error highlighting"""
        
        # Create DataFrame for display
        key_data = []
        for i in range(len(alice_key)):
            key_data.append({
                'Position': i,
                'Alice': alice_key[i],
                'Bob': bob_key[i],
                'Error': '❌' if errors[i] else '✅'
            })
        
        # Display in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Alice's Key:**")
            alice_str = ''.join(map(str, alice_key))
            st.code(alice_str, language=None)
        
        with col2:
            st.write("**Bob's Key:**")
            bob_str = ''.join(map(str, bob_key))
            st.code(bob_str, language=None)
        
        # Error analysis
        n_errors = np.sum(errors)
        st.metric("Initial Errors", f"{n_errors} / {len(alice_key)} ({n_errors/len(alice_key):.1%})")
    
    def hamming_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Dict:
        """Simulate Hamming code error correction"""
        
        # Simple Hamming (7,4) code simulation
        def encode_hamming_block(data_bits):
            """Encode 4 data bits using Hamming (7,4) code"""
            if len(data_bits) != 4:
                return data_bits  # Return as-is if not proper length
            
            d1, d2, d3, d4 = data_bits
            
            # Calculate parity bits
            p1 = d1 ^ d2 ^ d4
            p2 = d1 ^ d3 ^ d4  
            p3 = d2 ^ d3 ^ d4
            
            return np.array([p1, p2, d1, p3, d2, d3, d4])
        
        def decode_hamming_block(received_bits):
            """Decode Hamming (7,4) code and correct single errors"""
            if len(received_bits) != 7:
                return received_bits[:4]  # Return first 4 bits if not proper length
            
            p1, p2, d1, p3, d2, d3, d4 = received_bits
            
            # Calculate syndrome
            s1 = p1 ^ d1 ^ d2 ^ d4
            s2 = p2 ^ d1 ^ d3 ^ d4
            s3 = p3 ^ d2 ^ d3 ^ d4
            
            syndrome = s1 + 2*s2 + 4*s3
            
            # Correct error if syndrome != 0
            corrected = received_bits.copy()
            if syndrome != 0 and syndrome <= 7:
                corrected[syndrome - 1] = 1 - corrected[syndrome - 1]
            
            # Extract data bits
            return corrected[[2, 4, 5, 6]]
        
        # Pad keys to multiple of 4
        pad_length = (4 - len(alice_key) % 4) % 4
        alice_padded = np.concatenate([alice_key, np.zeros(pad_length, dtype=int)])
        bob_padded = np.concatenate([bob_key, np.zeros(pad_length, dtype=int)])
        
        # Process in blocks of 4
        alice_corrected = []
        bob_corrected = []
        communication_bits = 0
        
        for i in range(0, len(alice_padded), 4):
            alice_block = alice_padded[i:i+4]
            bob_block = bob_padded[i:i+4]
            
            # Alice encodes and sends parity information
            alice_encoded = encode_hamming_block(alice_block)
            parity_info = alice_encoded[[0, 1, 3]]  # Send parity bits
            communication_bits += 3
            
            # Bob uses parity info to correct his block
            bob_encoded = np.array([parity_info[0], parity_info[1], bob_block[0], 
                                  parity_info[2], bob_block[1], bob_block[2], bob_block[3]])
            
            bob_corrected_block = decode_hamming_block(bob_encoded)
            
            alice_corrected.extend(alice_block)
            bob_corrected.extend(bob_corrected_block)
        
        # Remove padding
        alice_final = np.array(alice_corrected[:len(alice_key)])
        bob_final = np.array(bob_corrected[:len(bob_key)])
        
        # Calculate statistics
        initial_errors = np.sum(alice_key != bob_key)
        final_errors = np.sum(alice_final != bob_final)
        
        return {
            'method': 'Hamming Code',
            'initial_errors': initial_errors,
            'final_errors': final_errors,
            'communication_bits': communication_bits,
            'alice_corrected': alice_final,
            'bob_corrected': bob_final,
            'efficiency': communication_bits / len(alice_key)
        }
    
    def cascade_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Dict:
        """Simulate CASCADE protocol"""
        
        alice_working = alice_key.copy()
        bob_working = bob_key.copy()
        
        communication_bits = 0
        iterations = 0
        max_iterations = 4
        
        # Initial error estimation
        initial_errors = np.sum(alice_working != bob_working)
        current_errors = initial_errors
        
        while current_errors > 0 and iterations < max_iterations:
            iterations += 1
            
            # Determine block size (adaptive)
            if iterations == 1:
                block_size = max(1, len(alice_working) // max(1, current_errors))
            else:
                block_size = max(1, block_size // 2)
            
            # Binary search for errors
            corrected_positions = []
            
            for start in range(0, len(alice_working), block_size):
                end = min(start + block_size, len(alice_working))
                
                # Alice sends parity of block
                alice_parity = np.sum(alice_working[start:end]) % 2
                bob_parity = np.sum(bob_working[start:end]) % 2
                communication_bits += 1
                
                if alice_parity != bob_parity:
                    # Binary search within block
                    error_pos = self.binary_search_error(
                        alice_working[start:end], 
                        bob_working[start:end]
                    )
                    
                    if error_pos is not None:
                        global_pos = start + error_pos
                        bob_working[global_pos] = 1 - bob_working[global_pos]
                        corrected_positions.append(global_pos)
                        communication_bits += int(np.log2(block_size)) + 1
            
            # Update error count
            current_errors = np.sum(alice_working != bob_working)
        
        return {
            'method': 'CASCADE',
            'initial_errors': initial_errors,
            'final_errors': current_errors,
            'communication_bits': communication_bits,
            'iterations': iterations,
            'alice_corrected': alice_working,
            'bob_corrected': bob_working,
            'efficiency': communication_bits / len(alice_key)
        }
    
    def binary_search_error(self, alice_block: np.ndarray, bob_block: np.ndarray) -> int:
        """Binary search for error within a block"""
        if len(alice_block) == 1:
            return 0 if alice_block[0] != bob_block[0] else None
        
        mid = len(alice_block) // 2
        
        # Check first half
        alice_parity_first = np.sum(alice_block[:mid]) % 2
        bob_parity_first = np.sum(bob_block[:mid]) % 2
        
        if alice_parity_first != bob_parity_first:
            # Error in first half
            sub_error = self.binary_search_error(alice_block[:mid], bob_block[:mid])
            return sub_error if sub_error is not None else None
        else:
            # Error in second half
            sub_error = self.binary_search_error(alice_block[mid:], bob_block[mid:])
            return mid + sub_error if sub_error is not None else None
    
    def parity_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Dict:
        """Simple parity-based error correction"""
        
        alice_working = alice_key.copy()
        bob_working = bob_key.copy()
        communication_bits = 0
        
        # Check overall parity
        alice_parity = np.sum(alice_working) % 2
        bob_parity = np.sum(bob_working) % 2
        communication_bits += 1
        
        if alice_parity != bob_parity:
            # Find error using bisection method
            block_size = len(alice_working) // 2
            
            while block_size >= 1:
                # Check first half
                alice_half_parity = np.sum(alice_working[:block_size]) % 2
                bob_half_parity = np.sum(bob_working[:block_size]) % 2
                communication_bits += 1
                
                if alice_half_parity != bob_half_parity:
                    # Error in first half
                    alice_working = alice_working[:block_size]
                    bob_working = bob_working[:block_size]
                else:
                    # Error in second half
                    alice_working = alice_working[block_size:]
                    bob_working = bob_working[block_size:]
                
                block_size = len(alice_working) // 2
            
            # Flip the erroneous bit
            if len(bob_working) == 1:
                bob_working[0] = 1 - bob_working[0]
        
        initial_errors = np.sum(alice_key != bob_key)
        final_errors = np.sum(alice_key != bob_key)  # Simple method doesn't guarantee correction
        
        return {
            'method': 'Parity Check',
            'initial_errors': initial_errors,
            'final_errors': final_errors,
            'communication_bits': communication_bits,
            'alice_corrected': alice_key,
            'bob_corrected': bob_key,
            'efficiency': communication_bits / len(alice_key)
        }
    
    def basic_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Dict:
        """Basic error correction (placeholder)"""
        
        initial_errors = np.sum(alice_key != bob_key)
        
        return {
            'method': 'Basic',
            'initial_errors': initial_errors,
            'final_errors': initial_errors,
            'communication_bits': 0,
            'alice_corrected': alice_key,
            'bob_corrected': bob_key,
            'efficiency': 0
        }
    
    def display_correction_results(self, results: Dict, method: str):
        """Display error correction results"""
        
        st.subheader(f"🔧 {method} Correction Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Initial Errors", results['initial_errors'])
        
        with col2:
            st.metric("Final Errors", results['final_errors'])
        
        with col3:
            st.metric("Communication Bits", results['communication_bits'])
        
        with col4:
            st.metric("Efficiency", f"{results['efficiency']:.2f}")
        
        # Success analysis
        if results['final_errors'] == 0:
            st.success("✅ Error correction successful!")
        elif results['final_errors'] < results['initial_errors']:
            st.warning(f"⚠️ Partial correction: {results['initial_errors'] - results['final_errors']} errors fixed")
        else:
            st.error("❌ Error correction failed")
        
        # Visualization
        self.create_correction_comparison(results)
    
    def create_correction_comparison(self, results: Dict):
        """Create before/after comparison visualization"""
        
        alice_corrected = results['alice_corrected']
        bob_corrected = results['bob_corrected']
        
        if len(alice_corrected) > 0:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Before Correction', 'After Correction'),
                vertical_spacing=0.15
            )
            
            positions = list(range(len(alice_corrected)))
            
            # Before correction (using original error pattern)
            initial_errors = results['initial_errors']
            before_errors = np.random.choice(positions, min(initial_errors, len(positions)), replace=False)
            
            fig.add_trace(go.Scatter(
                x=positions,
                y=alice_corrected,
                mode='markers',
                name="Alice",
                marker=dict(color='blue', size=8),
                showlegend=True
            ), row=1, col=1)
            
            # Simulate Bob's original errors
            bob_before = alice_corrected.copy()
            bob_before[before_errors] = 1 - bob_before[before_errors]
            
            fig.add_trace(go.Scatter(
                x=positions,
                y=bob_before,
                mode='markers',
                name="Bob (Before)",
                marker=dict(color='red', size=8),
                showlegend=True
            ), row=1, col=1)
            
            # After correction
            fig.add_trace(go.Scatter(
                x=positions,
                y=alice_corrected,
                mode='markers',
                name="Alice",
                marker=dict(color='blue', size=8),
                showlegend=False
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=positions,
                y=bob_corrected,
                mode='markers',
                name="Bob (After)",
                marker=dict(color='green', size=8),
                showlegend=True
            ), row=2, col=1)
            
            # Highlight remaining errors
            final_errors = alice_corrected != bob_corrected
            if np.any(final_errors):
                error_positions = np.array(positions)[final_errors]
                error_values = alice_corrected[final_errors]
                
                fig.add_trace(go.Scatter(
                    x=error_positions,
                    y=error_values,
                    mode='markers',
                    name='Remaining Errors',
                    marker=dict(color='red', size=15, symbol='x'),
                    showlegend=True
                ), row=2, col=1)
            
            fig.update_layout(
                title='Error Correction Comparison',
                height=500
            )
            
            for i in range(1, 3):
                fig.update_xaxes(title_text="Bit Position", row=i, col=1)
                fig.update_yaxes(title_text="Bit Value", row=i, col=1, tickvals=[0, 1])
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_correction_visualization(self):
        """Render error correction method visualization"""
        st.subheader("📊 Error Correction Overview")
        
        # Comparison table
        methods_data = {
            'Method': ['Hamming Code', 'CASCADE', 'Parity Check', 'BCH Code'],
            'Efficiency': ['Medium', 'High', 'Low', 'High'],
            'Complexity': ['Low', 'Medium', 'Very Low', 'High'],
            'Error Capability': ['Single', 'Multiple', 'Single', 'Multiple'],
            'Communication Cost': ['3 bits/4 data', 'Adaptive', '~log₂(n)', 'Variable']
        }
        
        import pandas as pd
        df = pd.DataFrame(methods_data)
        st.dataframe(df, use_container_width=True)
        
        # Error correction efficiency chart
        methods = ['Hamming', 'CASCADE', 'Parity', 'BCH']
        efficiency = [0.75, 0.95, 0.5, 0.85]
        communication_cost = [0.75, 0.3, 0.2, 0.6]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=methods,
            y=efficiency,
            name='Correction Efficiency',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=methods,
            y=communication_cost,
            name='Communication Cost',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Error Correction Method Comparison',
            xaxis_title='Method',
            yaxis_title='Relative Performance',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_algorithm_comparison(self):
        """Compare different error correction algorithms"""
        st.header("📊 Algorithm Performance Comparison")
        
        if st.button("🧪 Run Comparison Study"):
            self.run_algorithm_comparison()
    
    def run_algorithm_comparison(self):
        """Run comprehensive algorithm comparison"""
        
        # Test parameters
        key_lengths = [20, 50, 100]
        error_rates = [0.05, 0.1, 0.15, 0.2]
        methods = ['Hamming Code', 'CASCADE', 'Parity Check']
        
        results_summary = []
        
        for key_length in key_lengths:
            for error_rate in error_rates:
                for method in methods:
                    # Generate test case
                    alice_key = np.random.randint(0, 2, key_length)
                    bob_key = alice_key.copy()
                    error_positions = np.random.random(key_length) < error_rate
                    bob_key[error_positions] = 1 - bob_key[error_positions]
                    
                    # Apply correction
                    if method == "Hamming Code":
                        results = self.hamming_correction(alice_key, bob_key)
                    elif method == "CASCADE":
                        results = self.cascade_correction(alice_key, bob_key)
                    else:
                        results = self.parity_correction(alice_key, bob_key)
                    
                    # Store results
                    success_rate = 1 - (results['final_errors'] / max(1, results['initial_errors']))
                    
                    results_summary.append({
                        'Key Length': key_length,
                        'Error Rate': error_rate,
                        'Method': method,
                        'Success Rate': success_rate,
                        'Efficiency': results['efficiency'],
                        'Communication Cost': results['communication_bits']
                    })
        
        # Convert to DataFrame and display
        import pandas as pd
        df = pd.DataFrame(results_summary)
        
        # Group by method and show average performance
        method_performance = df.groupby('Method').agg({
            'Success Rate': 'mean',
            'Efficiency': 'mean', 
            'Communication Cost': 'mean'
        }).round(3)
        
        st.subheader("📈 Average Performance Summary")
        st.dataframe(method_performance)
        
        # Create performance visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Success Rate vs Error Rate', 'Communication Cost vs Key Length',
                          'Method Efficiency', 'Success Rate Distribution')
        )
        
        # Success rate vs error rate
        for method in methods:
            method_data = df[df['Method'] == method]
            fig.add_trace(go.Scatter(
                x=method_data['Error Rate'],
                y=method_data['Success Rate'],
                mode='markers+lines',
                name=method,
                showlegend=True
            ), row=1, col=1)
        
        # Communication cost vs key length
        for method in methods:
            method_data = df[df['Method'] == method]
            fig.add_trace(go.Scatter(
                x=method_data['Key Length'],
                y=method_data['Communication Cost'],
                mode='markers+lines',
                name=method,
                showlegend=False
            ), row=1, col=2)
        
        # Method efficiency (bar chart)
        avg_efficiency = df.groupby('Method')['Efficiency'].mean()
        fig.add_trace(go.Bar(
            x=list(avg_efficiency.index),
            y=list(avg_efficiency.values),
            name='Efficiency',
            showlegend=False
        ), row=2, col=1)
        
        # Success rate distribution (box plot)
        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method]
            fig.add_trace(go.Box(
                y=method_data['Success Rate'],
                name=method,
                showlegend=False
            ), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Error Correction Algorithm Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_cascade_protocol(self):
        """Detailed CASCADE protocol explanation"""
        st.header("🔄 CASCADE Protocol Deep Dive")
        
        st.markdown("""
        ### CASCADE Algorithm Overview
        
        CASCADE is an interactive error correction protocol specifically designed for quantum key distribution:
        
        1. **Adaptive Block Size**: Starts with blocks sized according to estimated error rate
        2. **Binary Search**: Uses binary search to locate errors within blocks
        3. **Multiple Passes**: Repeats with smaller blocks to catch remaining errors
        4. **Information Reconciliation**: Minimizes information leaked to eavesdropper
        """)
        
        # CASCADE simulation
        if st.button("🎬 Run CASCADE Animation"):
            self.cascade_animation()
        
        # CASCADE parameters
        st.subheader("⚙️ CASCADE Parameters")
        
        with st.expander("Parameter Analysis"):
            initial_error_rate = st.slider("Initial Error Rate", 0.01, 0.25, 0.1)
            key_length = st.slider("Key Length", 50, 200, 100)
            
            # Calculate optimal parameters
            optimal_block_size = max(1, int(0.73 / initial_error_rate))
            estimated_rounds = int(np.log2(key_length / optimal_block_size)) + 1
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Optimal Initial Block Size", optimal_block_size)
                st.metric("Estimated Rounds", estimated_rounds)
            
            with col2:
                estimated_communication = key_length * initial_error_rate * np.log2(key_length)
                st.metric("Est. Communication (bits)", f"{estimated_communication:.0f}")
                st.metric("Communication Ratio", f"{estimated_communication/key_length:.2f}")
    
    def cascade_animation(self):
        """Animate CASCADE protocol steps"""
        st.subheader("🎬 CASCADE Protocol Animation")
        
        # Simple animation simulation
        steps = [
            "1. **Initialize**: Alice and Bob have correlated keys with errors",
            "2. **Block Division**: Divide keys into blocks based on error rate estimate", 
            "3. **Parity Check**: Alice sends parity of each block to Bob",
            "4. **Error Detection**: Bob compares his parity with Alice's",
            "5. **Binary Search**: For blocks with parity mismatch, use binary search",
            "6. **Error Correction**: Bob flips the erroneous bit",
            "7. **Iteration**: Repeat with smaller blocks and shuffled positions",
            "8. **Termination**: Stop when no more errors are detected"
        ]
        
        for i, step in enumerate(steps):
            st.write(step)
            if i < len(steps) - 1:
                st.write("⬇️")
        
        st.info("💡 Each iteration roughly halves the block size, improving error detection precision")
    
    def render_theory_practice(self):
        """Theory and practical considerations"""
        st.header("📚 Error Correction Theory & Practice")
        
        theory_tabs = st.tabs(["Information Theory", "Practical Limits", "Security Impact", "Future Directions"])
        
        with theory_tabs[0]:
            st.subheader("📊 Information Theory Foundations")
            st.markdown("""
            ### Shannon's Channel Coding Theorem
            - **Channel Capacity**: C = 1 - H(p) where H(p) is binary entropy
            - **Error Correction Bound**: Rate ≤ C for reliable communication
            - **Practical Codes**: Approach but don't achieve Shannon limit
            
            ### Information Reconciliation
            - **Leaked Information**: I(Alice; Eve) ≤ communication overhead
            - **Privacy Amplification**: Remove leaked information using hash functions
            - **Trade-off**: Efficiency vs. security
            """)
        
        with theory_tabs[1]:
            st.subheader("⚠️ Practical Limitations")
            st.markdown("""
            ### Implementation Challenges
            - **Finite Key Length**: Statistical fluctuations in short keys
            - **Communication Errors**: Classical channel also has errors
            - **Computational Complexity**: Real-time processing requirements
            
            ### Performance Factors
            - **Initial Error Rate**: Higher rates require more communication
            - **Block Size Selection**: Trade-off between efficiency and success rate
            - **Termination Criteria**: When to stop the correction process
            """)
        
        with theory_tabs[2]:
            st.subheader("🔒 Security Implications")
            st.markdown("""
            ### Information Leakage
            - **Parity Bits**: Each parity check leaks 1 bit of information
            - **Binary Search**: Reveals position information about errors
            - **Statistical Analysis**: Pattern in communication can reveal key properties
            
            ### Countermeasures
            - **Privacy Amplification**: Hash the corrected key to remove leaked info
            - **Randomization**: Shuffle key positions between rounds
            - **Conservative Estimation**: Overestimate information leakage
            """)
        
        with theory_tabs[3]:
            st.subheader("🚀 Future Directions")
            st.markdown("""
            ### Advanced Techniques
            - **LDPC Codes**: Low-Density Parity-Check codes for better efficiency
            - **Polar Codes**: Achieve channel capacity in some scenarios
            - **Quantum Error Correction**: Direct quantum error correction methods
            
            ### Research Areas
            - **Rate-Adaptive Protocols**: Adjust to varying channel conditions
            - **Network QKD**: Error correction in quantum networks
            - **Post-Quantum Integration**: Hybrid classical-quantum approaches
            """)

def main():
    """Main function for error correction module"""
    module = ErrorCorrectionModule()
    module.main()
