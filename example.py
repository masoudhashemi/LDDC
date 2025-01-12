import os

from src.clustering import TextClustererSplit
from src.embeddings import TextEmbedder
from src.llm_client import LLMClient
from src.visualizer import ClusterVisualizer


def main():
    # Ensure you have set your OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
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
    llm_client = LLMClient()
    clusterer = TextClustererSplit(llm_client, similarity_threshold=0.3)

    # initial clustering
    initial_labels = clusterer.fit_transform(embeddings, texts)
    # Print results
    print("\nInitial Clustering Results:")
    for text, label in zip(texts, initial_labels):
        print(f"Text: {text:<50} Cluster: {label}")

    refined_labels = clusterer.refine_clusters(texts, max_iterations=2)
    print("\nRefined Clustering Results:")
    for text, label in zip(texts, refined_labels):
        print(f"Text: {text:<50} Cluster: {label}")

    # Visualize clusters
    visualizer = ClusterVisualizer()
    visualizer.visualize_clusters(embeddings, refined_labels, texts=texts)
    visualizer.show()


if __name__ == "__main__":
    main()
