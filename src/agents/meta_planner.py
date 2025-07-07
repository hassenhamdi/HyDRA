# src/agents/meta_planner.py
import yaml
import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

class MetaPlannerAgent:
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=gemini_api_key, temperature=0.0)
        with open("configs/agents.yaml", 'r') as f:
            self.prompt = PromptTemplate.from_template(yaml.safe_load(f)['meta_planner']['planning_prompt'])

    def generate_plan(self, query: str) -> list[str]:
        prompt = self.prompt.format(query=query)
        response = self.llm.invoke(prompt).content
        try:
            # Clean the response to ensure it's valid JSON
            clean_response = response.strip().replace("```json", "").replace("```", "")
            plan = json.loads(clean_response)
            return plan if isinstance(plan, list) else [query]
        except json.JSONDecodeError:
            print(f"Warning: Meta-Planner failed to generate a valid JSON plan. Falling back to a single task.")
            return [query]