import os

import yaml

from src.clustering import TextClustererSplit
from src.embeddings import TextEmbedder
from src.llm_client import LLMClient
from src.visualizer import ClusterVisualizer


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Load configuration
    config = load_config()

    # Ensure you have set your OpenAI API key
    if not os.getenv("OPENAI_API_KEY") and not config["model"]["api_key"]:
        print("Please set your OPENAI_API_KEY environment variable or in config.yaml")
        return

    # Sample texts
    texts = [
        "Machine learning is fascinating",
        "AI is changing the world",
        "Deep learning revolutionizes AI",
        "The weather is nice today",
        "It's a sunny day outside",
        "Snow is expected today",
        "Python is a great programming language",
        "I love coding in Python",
        "Programming is fun",
        "Machine learning uses deep neural networks",
    ]

    # Get embeddings
    embedder = TextEmbedder()
    embeddings = embedder.get_embeddings(texts)

    # Initialize LLM client and clusterer
    llm_client = LLMClient(config)
    clusterer = TextClustererSplit(llm_client, config=config["clustering"])

    # initial clustering
    initial_labels = clusterer.fit_transform(embeddings, texts)
    # Print results
    print("\nInitial Clustering Results:")
    for text, label in zip(texts, initial_labels):
        print(f"Text: {text:<50} Cluster: {label}")

    # Refine clusters
    refined_labels = clusterer.refine_clusters(texts)

    # Perform hierarchical clustering
    hierarchy_levels = clusterer.hierarchical_clustering(texts)

    # Print hierarchical results
    for level, labels in hierarchy_levels.items():
        print(f"\nHierarchy Level {level}:")
        for text, label in zip(texts, labels):
            print(f"Text: {text:<50} Cluster: {label}")

        # Print cluster summaries for this level
        summaries = clusterer.cluster_summaries
        print("\nCluster Summaries:")
        for cluster_id, summary in summaries.items():
            if cluster_id in set(labels):
                print(f"\nCluster {cluster_id}:")
                print(summary)

    # Visualize final level
    visualizer = ClusterVisualizer()
    visualizer.visualize_clusters(embeddings, list(hierarchy_levels.values())[-1], texts=texts)
    visualizer.show()


if __name__ == "__main__":
    main()
