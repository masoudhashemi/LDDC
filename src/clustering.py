from typing import Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class TextClustererSplit:
    """
    A DBSCAN/HDBSCAN-like clustering that uses:
    1) Embedding similarity (cosine similarity threshold)
    2) LLM checks to confirm or reject edge connections
    3) Optionally an iterative refinement that uses LLM-based cluster summaries and rules to reassign outliers.

    If an edge is rejected by LLM, the point may end up disconnected -> label = -1,
    ensuring it can be refined later.
    """

    DEFAULT_CONFIG = {
        "similarity_threshold": 0.8,
        "max_iterations": 2,
        "max_sample_texts": 10,
        "hierarchical_iterations": 3,
        "min_cluster_size": 2,
    }

    def __init__(self, llm_client, config: Dict = None):
        """
        Args:
            llm_client (LLMClient): The LLM client for checks & summarization
            config (Dict): Configuration options for clustering
        """
        self.config = config if config else self.DEFAULT_CONFIG
        self.similarity_threshold = self.config["similarity_threshold"]
        self.llm_client = llm_client
        self.checked_pairs = {}
        self.labels_ = None
        self.cluster_summaries = {}
        self.cluster_hierarchies = {}

    def fit_transform(self, embeddings, texts):
        """
        1) Identify candidate edges from embedding similarity >= similarity_threshold.
        2) For each candidate edge, call LLM to confirm (yes/no).
        3) Build connected components from the accepted edges. Singletons => label -1.
        4) Store cluster labels in self.labels_.

        Returns:
            np.ndarray: cluster labels, shape (n_samples,) -1 indicates outlier if a point remains disconnected
        """
        n_samples = len(texts)
        if n_samples == 0:
            self.labels_ = np.array([])
            return self.labels_

        print(f"\nStarting clustering with similarity threshold: {self.similarity_threshold}")

        sim_matrix = cosine_similarity(embeddings)
        approved_edges = [[] for _ in range(n_samples)]

        # 1) Collect candidate edges (i < j to avoid duplicates)
        candidate_pairs = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                similarity = sim_matrix[i, j]
                print(f"\nEmbedding similarity between texts {i} and {j}: {similarity:.3f}")
                if similarity >= self.similarity_threshold:
                    candidate_pairs.append((i, j))
                    print(f"Added as candidate pair")

        print(f"\nFound {len(candidate_pairs)} candidate pairs")

        # 2) Check each candidate pair with LLM
        for i, j in candidate_pairs:
            pair_key = (i, j)
            if pair_key not in self.checked_pairs:
                llm_response = self.llm_client.check_similarity(texts[i], texts[j])
                decision = self.llm_client.extract_answer(llm_response)
                self.checked_pairs[pair_key] = decision
                print(f"\nPair ({i}, {j}) - LLM Decision: {decision}")
            else:
                decision = self.checked_pairs[pair_key]
                print(f"\nPair ({i}, {j}) - Cached Decision: {decision}")

            if decision == "yes":
                approved_edges[i].append(j)
                approved_edges[j].append(i)
                print(f"Added edge between {i} and {j}")

        # Print approved edges
        print("\nApproved edges:")
        for i, edges in enumerate(approved_edges):
            if edges:
                print(f"Node {i} connected to: {edges}")

        # 3) Build connected components from approved_edges
        labels = np.full(n_samples, -1, dtype=int)
        visited = set()
        current_label = 0

        for node in range(n_samples):
            if node not in visited:
                # BFS/DFS if it has neighbors
                neighbors = approved_edges[node]
                if len(neighbors) == 0:
                    # This node is a singleton => stays -1
                    visited.add(node)
                    continue

                # Mark a cluster
                queue = [node]
                visited.add(node)
                labels[node] = current_label

                while queue:
                    cur = queue.pop(0)
                    for neigh in approved_edges[cur]:
                        if neigh not in visited:
                            visited.add(neigh)
                            labels[neigh] = current_label
                            queue.append(neigh)

                current_label += 1

        self.labels_ = labels
        return labels

    def refine_clusters(self, texts, max_iterations=None):
        """
        Iteratively:
        1) Summarize each cluster using LLM
        2) Try to assign outliers to the best-fitting cluster based on summary
        3) Re-summarize if you add new points, and repeat
        """
        max_iterations = self.DEFAULT_CONFIG["max_iterations"] if max_iterations is None else max_iterations

        if self.labels_ is None:
            raise ValueError("Must call fit_transform before refine_clusters.")

        labels = self.labels_
        n_samples = len(texts)

        # Utility: gather cluster memberships
        def get_clustered_texts(labels, texts):
            clustered = {}
            for i, lbl in enumerate(labels):
                if lbl != -1:
                    clustered.setdefault(lbl, []).append((i, texts[i]))
            return clustered

        # Utility: identify outliers
        def get_outliers(labels, texts):
            return [(i, texts[i]) for i, lbl in enumerate(labels) if lbl == -1]

        clustered_texts = get_clustered_texts(labels, texts)
        outliers = get_outliers(labels, texts)

        print("\nStarting cluster refinement:")
        print(f"Initial clusters: {clustered_texts}")
        print(f"Initial outliers: {outliers}")

        # Summaries stored as {cluster_id: summary_text}
        cluster_summaries = {}

        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}:")

            # 1) Summarize each cluster
            for c_id, c_members in clustered_texts.items():
                cluster_text_list = [txt for (_, txt) in c_members]
                summary_text = self.llm_client.summarize_cluster(cluster_text_list)
                cluster_summaries[c_id] = summary_text
                print(f"\nCluster {c_id} Summary: {summary_text}")

            if not outliers:
                print("No outliers to process. Stopping refinement.")
                break

            # 2) Attempt to assign outliers
            newly_assigned = []
            still_outliers_next = []

            for idx_out, text_o in outliers:
                print(f"\nProcessing outlier: {text_o}")

                assigned_cluster = None

                # Check each cluster's summary
                for c_id, summary in cluster_summaries.items():
                    print(f"\nChecking against cluster {c_id}")
                    print(f"Cluster summary: {summary}")
                    assignment_decision = self.llm_client.cluster_assignment_decision(text_o, summary)
                    print(f"Assignment decision: {assignment_decision}")

                    if assignment_decision:
                        assigned_cluster = c_id
                        break

                if assigned_cluster is not None:
                    newly_assigned.append((idx_out, text_o, assigned_cluster))
                    print(f"Assigned to cluster {assigned_cluster}")
                else:
                    still_outliers_next.append((idx_out, text_o))
                    print("Remained as outlier")

            if not newly_assigned:
                print("No new outliers were assigned in iteration", iteration)
                break

            # Merge newly assigned
            for idx_out, text_o, c_id in newly_assigned:
                clustered_texts[c_id].append((idx_out, text_o))
                labels[idx_out] = c_id

            outliers = still_outliers_next
            print(f"\nIteration {iteration} results:")
            print(f"Newly assigned: {newly_assigned}")
            print(f"Remaining outliers: {outliers}")

        self.labels_ = labels
        return labels

    def summarize_clusters(self, texts):
        """Summarize all current clusters."""
        clustered_texts = self._get_clustered_texts(texts)
        for c_id, c_members in clustered_texts.items():
            cluster_text_list = [txt for (_, txt) in c_members]
            summary_text = self.llm_client.summarize_cluster(cluster_text_list)
            self.cluster_summaries[c_id] = summary_text
        return self.cluster_summaries

    def hierarchical_clustering(self, texts):
        """
        Perform hierarchical clustering by iteratively combining clusters based on their summaries.
        """
        if self.labels_ is None:
            raise ValueError("Must call fit_transform before hierarchical_clustering")

        # Initial clustering and summaries
        current_labels = self.labels_.copy()
        hierarchy_levels = {0: current_labels.copy()}

        for level in range(self.config["hierarchical_iterations"]):
            # Get summaries for current clusters
            cluster_summaries = self.summarize_clusters(texts)

            # Skip if we have only one cluster
            unique_clusters = set(current_labels[current_labels != -1])
            if len(unique_clusters) <= 1:
                break

            # Compare each pair of cluster summaries
            merged_clusters = set()
            new_labels = current_labels.copy()
            cluster_pairs = [(c1, c2) for c1 in unique_clusters for c2 in unique_clusters if c1 < c2]

            for c1, c2 in cluster_pairs:
                if c1 in merged_clusters or c2 in merged_clusters:
                    continue

                # Use LLM to check if clusters should be merged based on their summaries
                should_merge = self.llm_client.cluster_assignment_decision(cluster_summaries[c1], cluster_summaries[c2])

                if should_merge:
                    # Merge clusters by updating labels
                    new_labels[new_labels == c2] = c1
                    merged_clusters.add(c2)

                    # Update hierarchy relationships
                    self.cluster_hierarchies.setdefault(level + 1, {}).setdefault(c1, []).append(c2)

            # Relabel clusters to be consecutive
            unique_new_labels = sorted(set(new_labels[new_labels != -1]))
            label_map = {old: new for new, old in enumerate(unique_new_labels)}
            current_labels = np.array([label_map.get(l, -1) for l in new_labels])

            # Store this level's clustering
            hierarchy_levels[level + 1] = current_labels.copy()

            if len(merged_clusters) == 0:
                print(f"No more clusters merged at level {level + 1}")
                break

        return hierarchy_levels

    def _get_clustered_texts(self, texts):
        """Utility function to gather cluster memberships."""
        clustered = {}
        for i, lbl in enumerate(self.labels_):
            if lbl != -1:
                clustered.setdefault(lbl, []).append((i, texts[i]))
        return clustered
