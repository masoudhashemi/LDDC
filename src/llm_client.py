import os
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


class LLMClient:
    def __init__(
        self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", rules_file: str = "src/domain_rules.txt"
    ):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.domain_rules = self._load_rules(rules_file)
        print(f"Loaded domain rules: {self.domain_rules}")

    def _load_rules(self, rules_file: str) -> str:
        """Load domain-specific rules from a file."""
        if not rules_file:
            return ""
        try:
            with open(rules_file, "r") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Warning: Could not load rules file: {e}")
            return ""

    def check_similarity_prompt(self, text1, text2):
        """
        Creates a prompt to ask the LLM: do these two texts belong to the same cluster?
        """
        domain_knowledge = f"Important domain-specific knowledge:\n{self.domain_rules}\n\n" if self.domain_rules else ""
        prompt = f"""Compare these two texts and determine if they belong to the same cluster.
Texts are in the same cluster only if they discuss the same core topic/concept. 
First identify the core topic/concept, then check if the texts discuss the same topic/concept.

Text 1: "{text1}"
Text 2: "{text2}"

Domain-specific knowledge that MUST be considered:
{domain_knowledge}

Response format -- provide the core topic of the two texts and finish with 'Answer: <yes/no>' 
yes, if they are in the same cluster, no otherwise:

- Core topic/concept:
    - text 1: <core topic of text 1>
    - text 2: <core topic of text 2>
- Answer: <yes/no>
"""
        return prompt

    def check_similarity(self, text1: str, text2: str) -> str:
        """
        Calls the LLM to see if text1 and text2 belong to the same cluster.

        Returns:
            str: the raw response text from the LLM, containing 'Answer: yes' or 'Answer: no'
        """
        prompt = self.check_similarity_prompt(text1, text2)
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=100
        )
        return response.choices[0].message.content.strip().lower()

    def extract_answer(self, llm_response):
        """
        Parse the LLM's response to extract 'yes' or 'no' after 'Answer:'.
        """
        # Lowercase and split by 'answer:'
        lower_resp = llm_response.lower()
        if "answer:" in lower_resp:
            answer_part = lower_resp.split("answer:")[1].strip()
            # Just in case there's extra punctuation
            if "yes" in answer_part:
                return "yes"
            elif "no" in answer_part:
                return "no"
        return "no"

    def summarize_cluster(self, cluster_texts: List[str]) -> str:
        """
        Use the LLM to create a 'summary' or 'rule' describing the cluster's overall topic.
        """
        if not cluster_texts:
            raise ValueError("Cannot summarize empty cluster")

        # Optionally sample if too large
        sample_texts = cluster_texts[:10]
        joined_texts = "\n- ".join(sample_texts)

        prompt = f"""We have a cluster of texts. Please provide a short, structured summary describing
the core topics or concepts that best represent the texts in this cluster.
Identify broad rules or key concepts that the cluster covers.

Cluster texts (sample up to 10 entries):
- {joined_texts}

Format your response as:

Summary: <a short phrase describing the cluster>
Rules:
- <optional bullet points or constraints>
"""
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=100
        )
        return response.choices[0].message.content.strip()

    def cluster_assignment_decision(self, outlier_text, cluster_summary):
        """
        Ask the LLM if an outlier text belongs to the cluster described by `cluster_summary`.
        """
        domain_knowledge = f"Important domain-specific knowledge:\n{self.domain_rules}\n\n" if self.domain_rules else ""
        prompt = f"""We have an outlier text that was not originally placed in any cluster.
Check if the text fits the following cluster summary.

Cluster summary: "{cluster_summary}"
Outlier text: "{outlier_text}"

Domain-specific knowledge that MUST be considered:
{domain_knowledge}

Answer yes or no: does this text clearly belong to the cluster described above?
"""
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=100
        )
        decision_text = response.choices[0].message.content.strip()
        return "yes" in decision_text
