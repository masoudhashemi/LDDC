import os

from openai import OpenAI


class LLMClient:
    def __init__(self, api_key=None):
        """
        Initialize LLM client

        Args:
            api_key: API key for OpenAI. If None, will look for OPENAI_API_KEY env variable
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def check_similarity(self, text1, text2):
        """
        Ask LLM to verify if two texts belong to the same cluster
        """
        prompt = f"""Compare these two texts and determine if they belong to the same cluster.
Texts are in the same cluster only if they discuss the same core topic/concept. First identify the core topic/concept, then check if the texts discuss the same topic/concept.
The core topic/concept is the most important part of the text and musst contain high level concepts and ideas represented in the text.
Kepp rainy separate from snowy days.

Text 1: "{text1}"
Text 2: "{text2}"

Response format -- provide the core topic of the two texts and finish with 'Answer: <yes/no>' yes, if they are in the same cluster, no otherwise:

- Core topic/concept:
    - text 1: <core topic of text 1>
    - text 2: <core topic of text 2>
- Answer: <yes/no>
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=100
        )

        return response.choices[0].message.content.strip().lower()

    def get_answer(self, llm_response):
        return llm_response.lower().split("answer:")[1].strip().lower()
