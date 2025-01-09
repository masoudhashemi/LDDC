import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TextClustererSplit:
    def __init__(self, llm_client, similarity_threshold=0.8):
        """
        A DBSCAN-like clustering with LLM verification that can also 'break' clusters.

        Args:
            llm_client: object with check_similarity(textA, textB) and get_answer(prompt)
            similarity_threshold: threshold for cosine similarity before LLM check
        """
        self.llm_client = llm_client
        self.similarity_threshold = similarity_threshold
        self.checked_pairs = {}

    def fit_transform(self, embeddings, texts):
        n_samples = len(texts)
        if n_samples == 0:
            return np.array([])

        sim_matrix = cosine_similarity(embeddings)

        # Maintain an adjacency list of "approved" edges
        # We'll only add edges after the LLM says "yes"
        approved_edges = [[] for _ in range(n_samples)]

        labels = np.full(n_samples, -1)
        visited = set()

        def get_cluster_components():
            # recomputing connected components after each BFS expansion
            comp_labels = np.full(n_samples, -1)
            c_id = 0
            for node in range(n_samples):
                if comp_labels[node] == -1:
                    # BFS to label this connected component
                    queue = [node]
                    comp_labels[node] = c_id
                    while queue:
                        cur = queue.pop(0)
                        for neigh in approved_edges[cur]:
                            if comp_labels[neigh] == -1:
                                comp_labels[neigh] = c_id
                                queue.append(neigh)
                    c_id += 1
            return comp_labels

        def try_approve_edge(i, j):
            """
            Checks LLM for similarity only if we haven't already checked (i, j).
            If LLM says 'yes', add i<->j to approved_edges;
            if 'no', ensure i<->j is removed.
            """
            
            # check if we've already checked this pair
            pair = (min(i, j), max(i, j))
            if pair in self.checked_pairs:
                decision = self.checked_pairs[pair]
            else:
                prompt = self.llm_client.check_similarity(texts[i], texts[j])
                decision = self.llm_client.get_answer(prompt).lower()
                self.checked_pairs[pair] = decision
                print("Text A: ", texts[i])
                print("Text B: ", texts[j])
                print("LLM score: ", prompt)
                print("Decision: ", decision)

            if "yes" in decision:
                if j not in approved_edges[i]:
                    approved_edges[i].append(j)
                if i not in approved_edges[j]:
                    approved_edges[j].append(i)
            else:
                if j in approved_edges[i]:
                    approved_edges[i].remove(j)
                if i in approved_edges[j]:
                    approved_edges[j].remove(i)

        # BFS-like approach to expand clusters
        for idx in range(n_samples):
            if idx in visited:
                continue
            queue = [idx]
            visited.add(idx)
            while queue:
                current = queue.pop(0)

                # Potential neighbors based on embedding similarity
                neighbors = np.where(sim_matrix[current] >= self.similarity_threshold)[0]
                for neigh in neighbors:
                    if neigh == current:
                        continue
                    # Attempt to approve the edge with LLM
                    try_approve_edge(current, neigh)

                    if neigh not in visited:
                        visited.add(neigh)
                        queue.append(neigh)

            # Recompute connected components after each BFS
            comp_labels = get_cluster_components()
            # Update final labels
            for i in range(n_samples):
                labels[i] = comp_labels[i]

        # Optionally renumber cluster labels from 0..K
        unique_comps = np.unique(labels)
        remap = {comp: i for i, comp in enumerate(unique_comps)}
        for i in range(n_samples):
            labels[i] = remap[labels[i]]

        return labels
