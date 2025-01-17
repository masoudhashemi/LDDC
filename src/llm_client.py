import os
from typing import Dict, List

import yaml
from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        config: Dict,
    ):
        self.client = OpenAI(api_key=config["model"].get("api_key") or os.getenv("OPENAI_API_KEY"))
        self.model = config["model"]["name"]
        self.domain_rules = self._load_rules(config["paths"]["domain_rules"])
        self.prompts = self._load_prompts(config["paths"]["prompts"])
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

    def _load_prompts(self, prompts_file: str) -> Dict[str, str]:
        """Load prompts from YAML file."""
        try:
            with open(prompts_file, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load prompts file: {e}")
            return {}

    def check_similarity_prompt(self, text1, text2):
        """
        Creates a prompt to ask the LLM: do these two texts belong to the same cluster?
        """
        domain_knowledge = f"Important domain-specific knowledge:\n{self.domain_rules}\n\n" if self.domain_rules else ""
        return self.prompts["similarity_check"].format(text1=text1, text2=text2, domain_knowledge=domain_knowledge)

    def check_similarity(self, text1: str, text2: str) -> str:
        """
        Calls the LLM to see if text1 and text2 belong to the same cluster.

        Returns:
            str: the raw response text from the LLM, containing 'Answer: yes' or 'Answer: no'
        """
        prompt = self.check_similarity_prompt(text1, text2)
        print("\nSimilarity Check:")
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Prompt: {prompt}")

        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=100
        )
        result = response.choices[0].message.content.strip().lower()
        print(f"LLM Response: {result}")
        return result

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
                answer = "yes"
            elif "no" in answer_part:
                answer = "no"
            else:
                answer = "no"
            print(f"Extracted answer: {answer}")
            return answer
        print("No 'Answer:' found in response, defaulting to 'no'")
        return "no"

    def summarize_cluster(self, cluster_texts: List[str]) -> str:
        """
        Enhanced cluster summarization that focuses on conceptual patterns and hypotheses.
        """
        if not cluster_texts:
            raise ValueError("Cannot summarize empty cluster")

        # Optionally sample if too large
        sample_texts = cluster_texts[:10]
        joined_texts = "\n- ".join(sample_texts)

        domain_knowledge = f"Domain-specific rules:\n{self.domain_rules}\n\n" if self.domain_rules else ""

        prompt = self.prompts["cluster_summary"].format(domain_knowledge=domain_knowledge, joined_texts=joined_texts)

        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=200
        )
        return response.choices[0].message.content.strip()

    def cluster_assignment_decision(self, outlier_text, cluster_summary):
        """
        Ask the LLM if an outlier text belongs to the cluster described by `cluster_summary`.
        """
        domain_knowledge = f"Important domain-specific knowledge:\n{self.domain_rules}\n\n" if self.domain_rules else ""
        prompt = self.prompts["cluster_assignment"].format(
            domain_knowledge=domain_knowledge, cluster_summary=cluster_summary, outlier_text=outlier_text
        )

        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=100
        )
        decision_text = response.choices[0].message.content.strip().lower()
        print(f"Decision text: {decision_text}")
        return "yes" in decision_text.split("answer:")[1].strip()
