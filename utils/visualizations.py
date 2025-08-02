"""
Complete visualization utilities for quantum cryptography education platform
Enhanced with unique chart keys to prevent Streamlit ID conflicts
All functions properly implemented and exported for BB84 protocol
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import streamlit as st
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import time
import uuid
import warnings
warnings.filterwarnings('ignore')

# Set style defaults
plt.style.use('default')
sns.set_palette("husl")

def generate_unique_key(base_name: str = "chart") -> str:
    """Generate unique key for Streamlit components to prevent ID conflicts"""
    timestamp = int(time.time() * 1000)
    unique_id = str(uuid.uuid4())[:8]
    return f"{base_name}_{timestamp}_{unique_id}"

class QuantumVisualization:
    """Base class for quantum visualizations with common styling and themes"""
    
    def __init__(self):
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2', 
            'success': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40',
            'sphere': 'rgba(100, 100, 255, 0.1)',
            'axes': 'rgba(0, 0, 0, 0.8)',
            'state_vector': '#ff4444',
            'quantum_blue': '#3498db',
            'quantum_red': '#e74c3c',
            'quantum_green': '#27ae60',
            'quantum_orange': '#f39c12',
            'quantum_purple': '#9b59b6'
        }
        
        self.plotly_theme = 'plotly_white'
        self.matplotlib_style = 'seaborn-v0_8'
    
    def get_color_scale(self, n_colors: int) -> List[str]:
        """Generate color scale for multi-element plots"""
        if n_colors <= 2:
            return [self.colors['primary'], self.colors['secondary']]
        elif n_colors <= 8:
            return px.colors.qualitative.Set3[:n_colors]
        else:
            return px.colors.qualitative.Plotly[:n_colors]
    
    def setup_matplotlib_style(self):
        """Setup matplotlib styling for consistent appearance"""
        try:
            plt.style.use(self.matplotlib_style)
        except:
            plt.style.use('default')
        
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'grid.alpha': 0.3
        })

class ProbabilityVisualization(QuantumVisualization):
    """Comprehensive probability distribution visualizations for quantum systems"""
    
    def create_measurement_probabilities(self, probabilities: Dict[str, float],
                                       theoretical: Optional[Dict[str, float]] = None,
                                       title: str = "Measurement Probabilities",
                                       show_percentages: bool = True) -> go.Figure:
        """
        Create interactive probability bar chart with optional theoretical comparison
        
        Args:
            probabilities: Experimental probabilities {'0': p0, '1': p1, ...}
            theoretical: Theoretical probabilities (optional)
            title: Chart title
            show_percentages: Whether to show percentage labels
        
        Returns:
            Plotly figure object
        """
        
        states = list(probabilities.keys())
        exp_probs = list(probabilities.values())
        
        fig = go.Figure()
        
        # Experimental probabilities
        fig.add_trace(go.Bar(
            x=states,
            y=exp_probs,
            name='Experimental',
            marker_color=self.colors['primary'],
            text=[f'{p:.3f} ({p*100:.1f}%)' if show_percentages else f'{p:.3f}' for p in exp_probs],
            textposition='auto',
            opacity=0.8,
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.4f}<br>Percentage: %{y:.1%}<extra></extra>'
        ))
        
        # Theoretical probabilities (if provided)
        if theoretical:
            theo_probs = [theoretical.get(state, 0) for state in states]
            fig.add_trace(go.Bar(
                x=states,
                y=theo_probs,
                name='Theoretical',
                marker_color=self.colors['secondary'],
                text=[f'{p:.3f} ({p*100:.1f}%)' if show_percentages else f'{p:.3f}' for p in theo_probs],
                textposition='auto',
                opacity=0.6,
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.4f}<br>Percentage: %{y:.1%}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color=self.colors['dark']),
                x=0.5
            ),
            xaxis_title='Quantum State',
            yaxis_title='Probability',
            barmode='group',
            yaxis=dict(
                range=[0, max(max(exp_probs), max(theoretical.values()) if theoretical else 1) * 1.1],
                tickformat='.3f'
            ),
            template=self.plotly_theme,
            height=400,
            showlegend=bool(theoretical),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_probability_evolution(self, time_steps: List[float],
                                   probability_data: Dict[str, List[float]],
                                   title: str = "Probability Evolution Over Time") -> go.Figure:
        """
        Create line plot showing probability evolution over time
        
        Args:
            time_steps: Time values
            probability_data: {'state': [prob_values_over_time]}
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        
        fig = go.Figure()
        colors = self.get_color_scale(len(probability_data))
        
        for i, (state, probs) in enumerate(probability_data.items()):
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=probs,
                mode='lines+markers',
                name=f'P(|{state}⟩)',
                line=dict(color=colors[i], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>P(|{state}⟩)</b><br>Time: %{{x}}<br>Probability: %{{y:.4f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title='Time',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            template=self.plotly_theme,
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_probability_histogram(self, measurement_results: List[int],
                                   bin_labels: Optional[List[str]] = None,
                                   title: str = "Measurement Results Distribution") -> go.Figure:
        """
        Create histogram of measurement results
        
        Args:
            measurement_results: List of measurement outcomes
            bin_labels: Optional labels for bins
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        
        fig = go.Figure()
        
        if bin_labels is None:
            bin_labels = [str(i) for i in sorted(set(measurement_results))]
        
        fig.add_trace(go.Histogram(
            x=measurement_results,
            nbinsx=len(bin_labels),
            name='Measurements',
            marker_color=self.colors['primary'],
            opacity=0.7,
            hovertemplate='<b>Outcome %{x}</b><br>Count: %{y}<br>Frequency: %{y/%{sum}:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title='Measurement Outcome',
            yaxis_title='Count',
            template=self.plotly_theme,
            height=400
        )
        
        return fig

class SecurityVisualization(QuantumVisualization):
    """Security analysis and cryptographic visualizations for quantum protocols"""
    
    def create_error_rate_analysis(self, error_rates: List[float],
                                 security_threshold: float = 0.11,
                                 title: str = "Quantum Bit Error Rate (QBER) Analysis",
                                 show_regions: bool = True) -> go.Figure:
        """
        Create comprehensive QBER analysis visualization for BB84 protocol
        
        Args:
            error_rates: List of measured error rates
            security_threshold: Security threshold (typically 11% for BB84)
            title: Chart title
            show_regions: Whether to show secure/insecure regions
        
        Returns:
            Plotly figure object
        """
        
        x_values = list(range(1, len(error_rates) + 1))
        
        fig = go.Figure()
        
        # Error rate line with enhanced styling
        fig.add_trace(go.Scatter(
            x=x_values,
            y=error_rates,
            mode='lines+markers',
            name='Measured QBER',
            line=dict(color=self.colors['primary'], width=4),
            marker=dict(size=10, color=self.colors['primary'], symbol='diamond'),
            hovertemplate='<b>Round %{x}</b><br>QBER: %{y:.4f} (%{y:.1%})<extra></extra>'
        ))
        
        # Security threshold line
        fig.add_hline(
            y=security_threshold,
            line_dash="dash",
            line_color=self.colors['error'],
            line_width=3,
            annotation_text=f"Security Threshold ({security_threshold:.1%})",
            annotation_position="top right",
            annotation_font_color=self.colors['error']
        )
        
        # Add colored regions if requested
        if show_regions:
            # Secure region
            fig.add_hrect(
                y0=0, y1=security_threshold,
                fillcolor=self.colors['success'], opacity=0.15,
                annotation_text="Secure Region", 
                annotation_position="top left",
                annotation_font_color=self.colors['success'],
                line_width=0
            )
            
            # Insecure region
            max_error = max(error_rates + [security_threshold * 1.5])
            fig.add_hrect(
                y0=security_threshold, y1=max_error,
                fillcolor=self.colors['error'], opacity=0.15,
                annotation_text="Insecure Region", 
                annotation_position="bottom left",
                annotation_font_color=self.colors['error'],
                line_width=0
            )
        
        # Calculate statistics
        avg_error = np.mean(error_rates)
        secure_rounds = sum(1 for rate in error_rates if rate <= security_threshold)
        security_ratio = secure_rounds / len(error_rates)
        
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>Average QBER: {avg_error:.3f} ({avg_error:.1%}) | Security Ratio: {security_ratio:.1%}</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Communication Round',
            yaxis_title='Quantum Bit Error Rate (QBER)',
            yaxis=dict(
                tickformat='.1%',
                range=[0, max(error_rates + [security_threshold * 1.2])]
            ),
            template=self.plotly_theme,
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_key_rate_analysis(self, distances: List[float],
                               key_rates: List[float],
                               title: str = "Secure Key Rate vs Distance") -> go.Figure:
        """
        Create key rate vs distance visualization for QKD protocols
        
        Args:
            distances: Communication distances (km)
            key_rates: Corresponding secure key rates (bits/second)
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        
        fig = go.Figure()
        
        # Key rate curve
        fig.add_trace(go.Scatter(
            x=distances,
            y=key_rates,
            mode='lines+markers',
            name='Secure Key Rate',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=8, color=self.colors['primary']),
            fill='tonexty',
            fillcolor=f'rgba(102, 126, 234, 0.1)',
            hovertemplate='<b>Distance: %{x:.1f} km</b><br>Key Rate: %{y:.2f} bits/sec<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=self.colors['dark'],
            annotation_text="No Secure Communication",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title='Distance (km)',
            yaxis_title='Secure Key Rate (bits/second)',
            template=self.plotly_theme,
            height=400
        )
        
        return fig
    
    def create_eavesdropping_detection(self, normal_errors: List[float],
                                     eavesdropping_errors: List[float],
                                     title: str = "Eavesdropping Detection Analysis") -> go.Figure:
        """
        Create visualization comparing normal vs eavesdropping error patterns
        
        Args:
            normal_errors: Error rates under normal conditions
            eavesdropping_errors: Error rates with eavesdropping
            title: Chart title
        
        Returns:
            Plotly figure object
        """
        
        fig = go.Figure()
        
        x_normal = list(range(1, len(normal_errors) + 1))
        x_eavesdrop = list(range(1, len(eavesdropping_errors) + 1))
        
        # Normal communication
        fig.add_trace(go.Scatter(
            x=x_normal,
            y=normal_errors,
            mode='lines+markers',
            name='Normal Communication',
            line=dict(color=self.colors['success'], width=3),
            marker=dict(size=8, color=self.colors['success'])
        ))
        
        # Eavesdropping present
        fig.add_trace(go.Scatter(
            x=x_eavesdrop,
            y=eavesdropping_errors,
            mode='lines+markers',
            name='Eavesdropping Detected',
            line=dict(color=self.colors['error'], width=3),
            marker=dict(size=8, color=self.colors['error'], symbol='x')
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title='Communication Round',
            yaxis_title='Error Rate',
            yaxis=dict(tickformat='.1%'),
            template=self.plotly_theme,
            height=400,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig

class BlochSphereVisualization(QuantumVisualization):
    """Advanced Bloch sphere visualizations for quantum state representation"""
    
    def create_bloch_sphere(self, theta: float, phi: float, 
                           title: str = "Quantum State on Bloch Sphere",
                           show_axes: bool = True,
                           show_grid: bool = True,
                           state_label: str = "|ψ⟩") -> go.Figure:
        """
        Create interactive 3D Bloch sphere with quantum state vector
        
        Args:
            theta: Polar angle in degrees (0-180)
            phi: Azimuthal angle in degrees (0-360)
            title: Plot title
            show_axes: Whether to show coordinate axes
            show_grid: Whether to show grid lines
            state_label: Label for the quantum state
        
        Returns:
            Plotly figure object
        """
        
        # Convert to radians
        theta_rad = np.pi * theta / 180
        phi_rad = np.pi * phi / 180
        
        # Calculate Cartesian coordinates
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)
        
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig = go.Figure()
        
        # Add Bloch sphere surface
        fig.add_trace(go.Surface(
            x=sphere_x, y=sphere_y, z=sphere_z,
            colorscale=[[0, 'rgba(100, 150, 255, 0.1)'], [1, 'rgba(100, 150, 255, 0.3)']],
            opacity=0.2,
            showscale=False,
            name="Bloch Sphere",
            hoverinfo='skip'
        ))
        
        # Add coordinate axes if requested
        if show_axes:
            axes_data = [
                # X-axis (red) - |+⟩/|-⟩ basis
                ([0, 1.2], [0, 0], [0, 0], self.colors['quantum_red'], '|+⟩'),
                ([-1.2, 0], [0, 0], [0, 0], self.colors['quantum_red'], '|-⟩'),
                # Y-axis (green) - |+i⟩/|-i⟩ basis
                ([0, 0], [0, 1.2], [0, 0], self.colors['quantum_green'], '|+i⟩'),
                ([0, 0], [-1.2, 0], [0, 0], self.colors['quantum_green'], '|-i⟩'),
                # Z-axis (blue) - |0⟩/|1⟩ basis
                ([0, 0], [0, 0], [0, 1.2], self.colors['quantum_blue'], '|0⟩'),
                ([0, 0], [0, 0], [-1.2, 0], self.colors['quantum_blue'], '|1⟩')
            ]
            
            for i, (x_coords, y_coords, z_coords, color, label) in enumerate(axes_data):
                fig.add_trace(go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='lines+text',
                    line=dict(color=color, width=5),
                    text=['', label],
                    textposition='middle center',
                    textfont=dict(size=14, color=color, family="Arial Black"),
                    showlegend=False,
                    name=f"Axis {i}",
                    hoverinfo='skip'
                ))
        
        # Add state vector
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            line=dict(color=self.colors['state_vector'], width=10),
            marker=dict(
                size=[8, 15],
                color=[self.colors['state_vector'], self.colors['quantum_orange']],
                symbol=['circle', 'diamond'],
                line=dict(width=2, color='white')
            ),
            name=f"State Vector {state_label}",
            showlegend=True,
            hovertemplate=f'<b>Quantum State {state_label}</b><br>θ = {theta:.1f}°<br>φ = {phi:.1f}°<br>Coordinates: ({x:.3f}, {y:.3f}, {z:.3f})<extra></extra>'
        ))
        
        # Add grid lines if requested
        if show_grid:
            # Equatorial circle
            t = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter3d(
                x=np.cos(t), y=np.sin(t), z=np.zeros_like(t),
                mode='lines',
                line=dict(color='rgba(128, 128, 128, 0.5)', width=2, dash='dash'),
                showlegend=False,
                name="Equator",
                hoverinfo='skip'
            ))
            
            # Prime meridian
            fig.add_trace(go.Scatter3d(
                x=np.cos(t), y=np.zeros_like(t), z=np.sin(t),
                mode='lines',
                line=dict(color='rgba(128, 128, 128, 0.3)', width=2, dash='dot'),
                showlegend=False,
                name="Meridian",
                hoverinfo='skip'
            ))
        
        # Layout configuration
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color=self.colors['dark']),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(range=[-1.5, 1.5], title='X', titlefont=dict(size=14)),
                yaxis=dict(range=[-1.5, 1.5], title='Y', titlefont=dict(size=14)),
                zaxis=dict(range=[-1.5, 1.5], title='Z', titlefont=dict(size=14)),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='rgba(248, 249, 250, 0.1)'
            ),
            width=700,
            height=700,
            margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            )
        )
        
        return fig

class Interactive3DBlochSphere(QuantumVisualization):
    """Enhanced Interactive 3D Bloch Sphere with advanced features"""
    
    def create_interactive_bloch_sphere(self, theta: float, phi: float, 
                                      title: str = "Interactive 3D Bloch Sphere",
                                      show_projections: bool = True,
                                      show_trajectory: bool = False,
                                      trajectory_points: Optional[List[Tuple[float, float]]] = None,
                                      show_quantum_gates: bool = False) -> go.Figure:
        """
        Create fully interactive 3D Bloch sphere with enhanced features
        """
        
        # Convert to radians
        theta_rad = np.pi * theta / 180
        phi_rad = np.pi * phi / 180
        
        # Calculate Cartesian coordinates for the state vector
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)
        
        # Create figure
        fig = go.Figure()
        
        # Add transparent sphere surface with ultra-high resolution
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=sphere_x, y=sphere_y, z=sphere_z,
            colorscale=[[0, 'rgba(100, 150, 255, 0.02)'], [0.5, 'rgba(100, 150, 255, 0.08)'], [1, 'rgba(100, 150, 255, 0.15)']],
            opacity=0.12,
            showscale=False,
            name="Bloch Sphere",
            hoverinfo='skip',
            lighting=dict(ambient=0.9, diffuse=0.8, specular=0.1)
        ))
        
        # Enhanced coordinate axes
        axis_length = 1.4
        axis_configs = [
            {'coords': [[-axis_length, axis_length], [0, 0], [0, 0]], 'color': '#e74c3c', 'width': 10, 'name': 'X-axis'},
            {'coords': [[0, 0], [-axis_length, axis_length], [0, 0]], 'color': '#27ae60', 'width': 10, 'name': 'Y-axis'},
            {'coords': [[0, 0], [0, 0], [-axis_length, axis_length]], 'color': '#3498db', 'width': 10, 'name': 'Z-axis'}
        ]
        
        for axis in axis_configs:
            fig.add_trace(go.Scatter3d(
                x=axis['coords'][0], y=axis['coords'][1], z=axis['coords'][2],
                mode='lines',
                line=dict(color=axis['color'], width=axis['width']),
                name=axis['name'],
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Enhanced axis labels
        label_distance = axis_length + 0.2
        labels = [
            {'pos': [label_distance, 0, 0], 'text': '|+⟩', 'color': '#e74c3c'},
            {'pos': [-label_distance, 0, 0], 'text': '|-⟩', 'color': '#e74c3c'},
            {'pos': [0, label_distance, 0], 'text': '|+i⟩', 'color': '#27ae60'},
            {'pos': [0, -label_distance, 0], 'text': '|-i⟩', 'color': '#27ae60'},
            {'pos': [0, 0, label_distance], 'text': '|0⟩', 'color': '#3498db'},
            {'pos': [0, 0, -label_distance], 'text': '|1⟩', 'color': '#3498db'}
        ]
        
        for label in labels:
            fig.add_trace(go.Scatter3d(
                x=[label['pos'][0]], y=[label['pos'][1]], z=[label['pos'][2]],
                mode='text',
                text=[label['text']],
                textfont=dict(size=22, color=label['color'], family="Arial Black"),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Enhanced reference circles
        t = np.linspace(0, 2*np.pi, 150)
        
        # Equator (XY plane)
        fig.add_trace(go.Scatter3d(
            x=np.cos(t), y=np.sin(t), z=np.zeros_like(t),
            mode='lines',
            line=dict(color='rgba(128, 128, 128, 0.8)', width=5, dash='dash'),
            name='Equator',
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Add trajectory if provided
        if show_trajectory and trajectory_points:
            traj_coords = []
            for t_theta, t_phi in trajectory_points:
                t_theta_rad = np.pi * t_theta / 180
                t_phi_rad = np.pi * t_phi / 180
                traj_x = np.sin(t_theta_rad) * np.cos(t_phi_rad)
                traj_y = np.sin(t_theta_rad) * np.sin(t_phi_rad)
                traj_z = np.cos(t_theta_rad)
                traj_coords.append((traj_x, traj_y, traj_z))
            
            if traj_coords:
                traj_x, traj_y, traj_z = zip(*traj_coords)
                fig.add_trace(go.Scatter3d(
                    x=traj_x, y=traj_y, z=traj_z,
                    mode='lines+markers',
                    line=dict(color='#f39c12', width=8, dash='dash'),
                    marker=dict(size=5, color='#f39c12', opacity=0.8),
                    name='State Evolution',
                    opacity=0.8
                ))
        
        # Ultra-enhanced state vector
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            line=dict(color=self.colors['state_vector'], width=15),
            marker=dict(
                size=[12, 30],
                color=[self.colors['state_vector'], '#ff6666'],
                symbol=['circle', 'diamond'],
                line=dict(width=4, color='white'),
                opacity=[0.9, 1.0]
            ),
            name=f"Quantum State |ψ⟩",
            hovertemplate=f"<b>Quantum State Vector</b><br>θ = {theta:.1f}° ({theta_rad:.3f} rad)<br>φ = {phi:.1f}° ({phi_rad:.3f} rad)<br>Cartesian: ({x:.3f}, {y:.3f}, {z:.3f})<br>Magnitude: {np.sqrt(x**2 + y**2 + z**2):.3f}<extra></extra>"
        ))
        
        # Add projection lines if requested
        if show_projections:
            projections = [
                {'coords': [[x, x], [y, y], [z, 0]], 'color': 'rgba(255, 99, 71, 0.8)', 'name': 'Z Projection'},
                {'coords': [[x, x], [y, 0], [z, z]], 'color': 'rgba(144, 238, 144, 0.8)', 'name': 'Y Projection'},
                {'coords': [[x, 0], [y, y], [z, z]], 'color': 'rgba(173, 216, 230, 0.8)', 'name': 'X Projection'}
            ]
            
            for proj in projections:
                fig.add_trace(go.Scatter3d(
                    x=proj['coords'][0], y=proj['coords'][1], z=proj['coords'][2],
                    mode='lines',
                    line=dict(color=proj['color'], width=6, dash='dot'),
                    name=proj['name'],
                    opacity=0.8,
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        # Ultra-enhanced layout
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub style='color: {self.colors['secondary']}; font-size: 16px;'>θ = {theta:.1f}°, φ = {phi:.1f}° | |r⃗| = {np.sqrt(x**2 + y**2 + z**2):.3f}</sub>",
                font=dict(size=26, color=self.colors['primary'], family="Arial Black"),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    range=[-2.0, 2.0],
                    title=dict(text='X', font=dict(size=18, color=self.colors['dark'])),
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    gridwidth=2,
                    showticklabels=True,
                    tickfont=dict(size=14)
                ),
                yaxis=dict(
                    range=[-2.0, 2.0],
                    title=dict(text='Y', font=dict(size=18, color=self.colors['dark'])),
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    gridwidth=2,
                    showticklabels=True,
                    tickfont=dict(size=14)
                ),
                zaxis=dict(
                    range=[-2.0, 2.0],
                    title=dict(text='Z', font=dict(size=18, color=self.colors['dark'])),
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    gridwidth=2,
                    showticklabels=True,
                    tickfont=dict(size=14)
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=2.2, y=2.2, z=2.2),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor='rgba(248, 249, 250, 0.05)'
            ),
            width=1100,
            height=900,
            margin=dict(l=0, r=0, t=140, b=0),
            showlegend=True,
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='rgba(100, 100, 100, 0.3)',
                borderwidth=2,
                font=dict(size=14)
            )
        )
        
        return fig
    
    def create_state_info_panel(self, theta: float, phi: float) -> Dict:
        """Create comprehensive state information panel"""
        
        theta_rad = np.pi * theta / 180
        phi_rad = np.pi * phi / 180
        
        # Calculate state vector components
        alpha = np.cos(theta_rad / 2)
        beta = np.exp(1j * phi_rad) * np.sin(theta_rad / 2)
        
        # Calculate Bloch vector coordinates
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)
        
        # Measurement probabilities
        prob_0 = np.abs(alpha) ** 2
        prob_1 = np.abs(beta) ** 2
        
        # Advanced quantum properties
        purity = 1.0
        von_neumann_entropy = 0.0
        
        return {
            'state_vector': {
                'alpha': alpha, 'beta': beta,
                'alpha_real': np.real(alpha), 'alpha_imag': np.imag(alpha),
                'beta_real': np.real(beta), 'beta_imag': np.imag(beta),
                'alpha_magnitude': np.abs(alpha), 'beta_magnitude': np.abs(beta)
            },
            'bloch_vector': {
                'x': x, 'y': y, 'z': z, 
                'magnitude': np.sqrt(x**2 + y**2 + z**2)
            },
            'angles': {
                'theta': theta, 'phi': phi, 
                'theta_rad': theta_rad, 'phi_rad': phi_rad
            },
            'probabilities': {'prob_0': prob_0, 'prob_1': prob_1},
            'phase': {
                'absolute': np.angle(beta), 
                'degrees': np.degrees(np.angle(beta))
            },
            'quantum_properties': {
                'purity': purity,
                'von_neumann_entropy': von_neumann_entropy,
                'expectation_values': {'sigma_x': x, 'sigma_y': y, 'sigma_z': z}
            }
        }

class ChannelVisualization(QuantumVisualization):
    """Specialized visualizations for quantum channel errors and noise analysis"""
    
    def create_channel_fidelity_plot(self, noise_levels: List[float],
                                   fidelities: List[float],
                                   title: str = "Channel Fidelity vs Noise Level") -> go.Figure:
        """
        Create channel fidelity visualization
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=noise_levels,
            y=fidelities,
            mode='lines+markers',
            name='Channel Fidelity',
            line=dict(color=self.colors['primary'], width=4),
            marker=dict(size=10, color=self.colors['primary']),
            hovertemplate='<b>Noise Level: %{x:.3f}</b><br>Fidelity: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title='Noise Level',
            yaxis_title='Channel Fidelity',
            yaxis=dict(range=[0, 1]),
            template=self.plotly_theme,
            height=400
        )
        
        return fig
    
    def create_noise_model_comparison(self, noise_models: Dict[str, List[float]],
                                    x_values: List[float],
                                    title: str = "Noise Model Comparison") -> go.Figure:
        """
        Compare different noise models
        """
        fig = go.Figure()
        colors = self.get_color_scale(len(noise_models))
        
        for i, (model_name, values) in enumerate(noise_models.items()):
            fig.add_trace(go.Scatter(
                x=x_values,
                y=values,
                mode='lines+markers',
                name=model_name,
                line=dict(color=colors[i], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title='Parameter',
            yaxis_title='Effect',
            template=self.plotly_theme,
            height=400,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig

def create_enhanced_interactive_lab():
    """Create comprehensive interactive quantum laboratory"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 3rem; border-radius: 25px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.8em;">🌐 Advanced 3D Bloch Sphere Laboratory</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.3em;">Comprehensive Real-time Quantum State Visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    bloch_viz = Interactive3DBlochSphere()
    
    st.sidebar.markdown("### 🎛️ Advanced Quantum State Controls")
    
    theta = st.sidebar.slider("**Polar Angle θ** (degrees)", 0, 180, 45, 1)
    phi = st.sidebar.slider("**Azimuthal Angle φ** (degrees)", 0, 360, 0, 1)
    
    main_col, info_col = st.columns([2.2, 1], gap="large")
    
    with main_col:
        fig = bloch_viz.create_interactive_bloch_sphere(theta, phi)
        st.plotly_chart(fig, use_container_width=True, key=generate_unique_key("enhanced_bloch_sphere"))
    
    with info_col:
        state_info = bloch_viz.create_state_info_panel(theta, phi)
        
        st.markdown("### 📊 Quantum Analysis")
        
        alpha = state_info['state_vector']['alpha']
        beta = state_info['state_vector']['beta']
        
        st.markdown(f"""
        **State Vector:**
        ```
        |ψ⟩ = ({alpha.real:.3f} + {alpha.imag:.3f}i)|0⟩ 
            + ({beta.real:.3f} + {beta.imag:.3f}i)|1⟩
        ```
        """)
        
        prob_0 = state_info['probabilities']['prob_0']
        prob_1 = state_info['probabilities']['prob_1']
        
        prob_fig = go.Figure(data=[
            go.Bar(x=['|0⟩', '|1⟩'], y=[prob_0, prob_1], 
                  marker_color=['#e74c3c', '#3498db'])
        ])
        prob_fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(prob_fig, use_container_width=True, key=generate_unique_key("enhanced_lab_probabilities"))
    
    return theta, phi, state_info

def safe_plotly_chart(fig: go.Figure, key_suffix: str = "", **kwargs):
    """Safe wrapper for plotly charts with guaranteed unique keys"""
    unique_key = generate_unique_key(f"chart_{key_suffix}")
    kwargs.pop('key', None)
    return st.plotly_chart(fig, key=unique_key, **kwargs)

# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN UTILITY FUNCTIONS - These are the functions your modules are importing
# ═══════════════════════════════════════════════════════════════════════════════════════

def plot_probabilities(probabilities: Dict[str, float], 
                      theoretical: Optional[Dict[str, float]] = None,
                      title: str = "Measurement Probabilities",
                      key_suffix: str = "",
                      **kwargs) -> go.Figure:
    """
    Main utility function to create and display probability charts
    
    Args:
        probabilities: Dictionary of probabilities {'0': p0, '1': p1, ...}
        theoretical: Optional theoretical probabilities for comparison
        title: Chart title
        key_suffix: Unique identifier for chart key
        **kwargs: Additional plotting arguments
    
    Returns:
        Plotly figure object
    """
    viz = ProbabilityVisualization()
    fig = viz.create_measurement_probabilities(
        probabilities=probabilities,
        theoretical=theoretical,
        title=title,
        **kwargs
    )
    
    # Auto-display if in Streamlit context
    try:
        if hasattr(st, 'plotly_chart'):
            unique_key = generate_unique_key(f"prob_{key_suffix}" if key_suffix else "probabilities")
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
    except:
        pass  # Not in Streamlit context
    
    return fig

def plot_error_analysis(error_rates: List[float], 
                       security_threshold: float = 0.11,
                       title: str = "Quantum Bit Error Rate Analysis",
                       key_suffix: str = "",
                       **kwargs) -> go.Figure:
    """
    Main utility function to create and display error rate analysis
    
    Args:
        error_rates: List of measured error rates
        security_threshold: Security threshold (default 11% for BB84)
        title: Chart title
        key_suffix: Unique identifier for chart key
        **kwargs: Additional plotting arguments
    
    Returns:
        Plotly figure object
    """
    viz = SecurityVisualization()
    fig = viz.create_error_rate_analysis(
        error_rates=error_rates,
        security_threshold=security_threshold,
        title=title,
        **kwargs
    )
    
    # Auto-display if in Streamlit context
    try:
        if hasattr(st, 'plotly_chart'):
            unique_key = generate_unique_key(f"error_{key_suffix}" if key_suffix else "error_analysis")
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
    except:
        pass  # Not in Streamlit context
    
    return fig

def plot_bloch_sphere(theta: float, phi: float, 
                     title: str = "Quantum State on Bloch Sphere",
                     key_suffix: str = "",
                     **kwargs) -> go.Figure:
    """
    Main utility function to create and display Bloch sphere
    
    Args:
        theta: Polar angle in degrees (0-180)
        phi: Azimuthal angle in degrees (0-360)
        title: Chart title
        key_suffix: Unique identifier for chart key
        **kwargs: Additional plotting arguments
    
    Returns:
        Plotly figure object
    """
    viz = BlochSphereVisualization()
    fig = viz.create_bloch_sphere(theta=theta, phi=phi, title=title, **kwargs)
    
    # Auto-display if in Streamlit context
    try:
        if hasattr(st, 'plotly_chart'):
            unique_key = generate_unique_key(f"bloch_{key_suffix}" if key_suffix else "bloch_sphere")
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
    except:
        pass  # Not in Streamlit context
    
    return fig

def plot_key_rate_analysis(distances: List[float], key_rates: List[float],
                          title: str = "Secure Key Rate vs Distance",
                          key_suffix: str = "",
                          **kwargs) -> go.Figure:
    """
    Utility function for key rate analysis plots
    """
    viz = SecurityVisualization()
    fig = viz.create_key_rate_analysis(distances, key_rates, title)
    
    try:
        if hasattr(st, 'plotly_chart'):
            unique_key = generate_unique_key(f"keyrate_{key_suffix}" if key_suffix else "key_rate")
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
    except:
        pass
    
    return fig

def plot_eavesdropping_detection(normal_errors: List[float], 
                               eavesdropping_errors: List[float],
                               title: str = "Eavesdropping Detection Analysis",
                               key_suffix: str = "",
                               **kwargs) -> go.Figure:
    """
    Utility function for eavesdropping detection plots
    """
    viz = SecurityVisualization()
    fig = viz.create_eavesdropping_detection(normal_errors, eavesdropping_errors, title)
    
    try:
        if hasattr(st, 'plotly_chart'):
            unique_key = generate_unique_key(f"eavesdrop_{key_suffix}" if key_suffix else "eavesdropping")
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
    except:
        pass
    
    return fig

def plot_channel_fidelity(noise_levels: List[float], fidelities: List[float],
                         title: str = "Channel Fidelity vs Noise Level",
                         key_suffix: str = "",
                         **kwargs) -> go.Figure:
    """
    Utility function for channel fidelity plots
    """
    viz = ChannelVisualization()
    fig = viz.create_channel_fidelity_plot(noise_levels, fidelities, title)
    
    try:
        if hasattr(st, 'plotly_chart'):
            unique_key = generate_unique_key(f"fidelity_{key_suffix}" if key_suffix else "channel_fidelity")
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
    except:
        pass
    
    return fig

def create_interactive_polarization_lab():
    """Quick function to create interactive lab"""
    return create_enhanced_interactive_lab()

# ═══════════════════════════════════════════════════════════════════════════════════════
# COMPLETE EXPORT LIST - All functions available for import
# ═══════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main visualization classes
    'QuantumVisualization',
    'BlochSphereVisualization', 
    'ProbabilityVisualization',
    'SecurityVisualization',
    'ChannelVisualization',
    'Interactive3DBlochSphere',
    
    # Primary utility functions (most commonly imported)
    'plot_probabilities',           # ← Main function for channel_errors
    'plot_error_analysis',          # ← Main function for channel_errors  
    'plot_bloch_sphere',           # ← Main function for polarization
    'plot_key_rate_analysis',      # ← For security analysis
    'plot_eavesdropping_detection', # ← For security analysis
    'plot_channel_fidelity',       # ← For channel analysis
    
    # Laboratory functions
    'create_enhanced_interactive_lab',
    'create_interactive_polarization_lab',
    
    # Helper utilities
    'safe_plotly_chart',
    'generate_unique_key'
]
