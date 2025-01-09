import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


class ClusterVisualizer:
    def __init__(self):
        """Initialize the ClusterVisualizer"""
        self.fig = None

    def visualize_clusters(self, embeddings, labels, texts=None):
        """
        Visualize clusters using PCA projection to 2D with interactive Plotly plot

        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            labels: numpy array of cluster labels
            texts: list of strings to display as hover text (optional)
        """
        # Project embeddings to 2D using PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        # Create a DataFrame for Plotly
        import pandas as pd

        df = pd.DataFrame(
            {
                "PC1": embeddings_2d[:, 0],
                "PC2": embeddings_2d[:, 1],
                "Cluster": [f"Cluster {label}" if label != -1 else "Noise" for label in labels],
            }
        )

        if texts is not None:
            df["Text"] = texts

        # Create interactive scatter plot
        self.fig = px.scatter(
            df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title="Cluster Visualization (PCA projection)",
            labels={"PC1": "First Principal Component", "PC2": "Second Principal Component"},
            color_discrete_sequence=px.colors.qualitative.Set1,
            opacity=0.7,
            hover_data=["Text"] if texts is not None else None,
        )

        # Add text annotations
        if texts is not None:
            self.fig.add_trace(
                go.Scatter(
                    x=df["PC1"],
                    y=df["PC2"],
                    mode="text",
                    text=[text[:30] + "..." if len(text) > 30 else text for text in texts],
                    textposition="top center",
                    showlegend=False,
                    textfont=dict(size=8),
                    hoverinfo="skip",
                )
            )

        # Update layout for better visualization
        self.fig.update_layout(
            plot_bgcolor="white",
            showlegend=True,
            legend_title_text="",
            title_x=0.5,
        )

    def show(self):
        """Display the plot"""
        if self.fig is not None:
            self.fig.show()
        else:
            raise ValueError("No visualization has been created yet. Call visualize_clusters first.")

    def save(self, filename):
        """Save the plot to a file"""
        if self.fig is not None:
            self.fig.write_html(filename)
        else:
            raise ValueError("No visualization has been created yet. Call visualize_clusters first.")
