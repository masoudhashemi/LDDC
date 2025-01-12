import os

from openai import OpenAI


class LLMClient:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def check_similarity_prompt(self, text1, text2):
        """
        Creates a prompt to ask the LLM: do these two texts belong to the same cluster?
        """
        prompt = f"""Compare these two texts and determine if they belong to the same cluster.
Texts are in the same cluster only if they discuss the same core topic/concept. 
First identify the core topic/concept, then check if the texts discuss the same topic/concept.

Note that snow is not about weather. Do not include weather in your answer.
Remember: "snow" can refer to an LLM model, not necessarily weather.

Text 1: "{text1}"
Text 2: "{text2}"

Response format -- provide the core topic of the two texts and finish with 'Answer: <yes/no>' 
yes, if they are in the same cluster, no otherwise:

- Core topic/concept:
    - text 1: <core topic of text 1>
    - text 2: <core topic of text 2>
- Answer: <yes/no>
"""
        return prompt

    def check_similarity(self, text1, text2):
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

    def summarize_cluster(self, cluster_texts):
        """
        Use the LLM to create a 'summary' or 'rule' describing the cluster's overall topic.
        """
        # Optionally sample if too large
        sample_texts = cluster_texts[:10]
        joined_texts = "\n- ".join(sample_texts)

        prompt = f"""
We have a cluster of texts. Please provide a short, structured summary describing
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
        prompt = f"""
Important domain-specific knowledge:
- 'snow' refers to an LLM model. Texts about snow should be placed in the AI cluster.

We have an outlier text that was not originally placed in any cluster.
Check if the text fits the following cluster summary.

Cluster summary: "{cluster_summary}"
Outlier text: "{outlier_text}"

Answer yes or no: does this text clearly belong to the cluster described above?
"""
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=100
        )
        decision_text = response.choices[0].message.content.strip()
        return "yes" in decision_text
