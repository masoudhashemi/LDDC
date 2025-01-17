# LLM-based clustering

This is a simple implementation of a clustering algorithm that uses LLMs to check if two texts are similar and to summarize clusters.

This uses a simple DBSCAN-like approach, but with LLM-based similarity checks based on a domain-specific rules.

The clustering is done in two steps:

1. First, a similarity check is done to check if two texts are similar.
2. Then, a cluster summary is done to summarize the cluster.

## Usage

```python
python example.py
```
