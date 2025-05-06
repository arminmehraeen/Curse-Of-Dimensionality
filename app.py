import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
import math
from typing import List, Tuple
import time

st.set_page_config(
    page_title="Curse of Dimensionality Explorer",
    page_icon="📊",
    layout="wide"
)

def calculate_distances(data: np.ndarray) -> Tuple[float, float, float]:
    """Calculate min, max, and mean distances between points."""
    distances = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distances.append(np.linalg.norm(data[i] - data[j]))
    return min(distances), max(distances), np.mean(distances)

def generate_random_points(n_points: int, n_dims: int) -> np.ndarray:
    """Generate random points in n-dimensional space."""
    return np.random.rand(n_points, n_dims)

def calculate_nearest_neighbors(data: np.ndarray, k: int = 5) -> List[float]:
    """Calculate k-nearest neighbor distances."""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    return distances[:, 1:].mean(axis=1)

def main():
    st.title("📊 Curse of Dimensionality Explorer")
    st.markdown("""
    This interactive tool demonstrates the curse of dimensionality by showing how distances between points
    behave in different dimensional spaces. The curse of dimensionality refers to various phenomena that
    arise when analyzing and organizing data in high-dimensional spaces.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")
        n_points = st.slider("Number of Points", 50, 1000, 200)
        max_dims = st.slider("Maximum Dimensions", 2, 100, 50)
        step_size = st.slider("Dimension Step Size", 1, 10, 2)
        k_neighbors = st.slider("K-Nearest Neighbors", 1, 10, 5)

    with st.spinner("Computing..."):
        dimensions = range(2, max_dims + 1, step_size)
        results = []
        
        for dim in dimensions:
            data = generate_random_points(n_points, dim)
            min_dist, max_dist, mean_dist = calculate_distances(data)
            nn_distances = calculate_nearest_neighbors(data, k_neighbors)
            
            results.append({
                'dimensions': dim,
                'min_distance': min_dist,
                'max_distance': max_dist,
                'mean_distance': mean_dist,
                'nn_mean_distance': np.mean(nn_distances),
                'distance_ratio': max_dist / min_dist if min_dist > 0 else 0
            })

        df = pd.DataFrame(results)

        # Plot 1: Distance Ratios
        fig1 = px.line(df, x='dimensions', y='distance_ratio',
                      title='Distance Ratio (Max/Min) vs Dimensions',
                      labels={'dimensions': 'Number of Dimensions',
                             'distance_ratio': 'Distance Ratio'})
        fig1.update_layout(showlegend=True)
        st.plotly_chart(fig1, use_container_width=True)

        # Plot 2: Distance Distributions
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['dimensions'], y=df['min_distance'],
                                 name='Min Distance', mode='lines+markers'))
        fig2.add_trace(go.Scatter(x=df['dimensions'], y=df['max_distance'],
                                 name='Max Distance', mode='lines+markers'))
        fig2.add_trace(go.Scatter(x=df['dimensions'], y=df['mean_distance'],
                                 name='Mean Distance', mode='lines+markers'))
        fig2.update_layout(title='Distance Metrics vs Dimensions',
                          xaxis_title='Number of Dimensions',
                          yaxis_title='Distance',
                          showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

        # Plot 3: K-NN Distances
        fig3 = px.line(df, x='dimensions', y='nn_mean_distance',
                      title='K-Nearest Neighbor Mean Distance vs Dimensions',
                      labels={'dimensions': 'Number of Dimensions',
                             'nn_mean_distance': 'Mean K-NN Distance'})
        fig3.update_layout(showlegend=True)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    ### Key Observations
    1. As dimensions increase, the ratio between maximum and minimum distances approaches 1
    2. The mean distance between points increases with dimensionality
    3. K-nearest neighbor distances become less meaningful in higher dimensions
    """)

if __name__ == "__main__":
    main() 