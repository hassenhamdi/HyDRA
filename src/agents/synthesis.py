# src/agents/synthesis.py
import yaml
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class SynthesisAgent:
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=gemini_api_key, temperature=0.3)
        with open("configs/agents.yaml", 'r') as f:
            self.prompt = PromptTemplate.from_template(yaml.safe_load(f)['synthesis_agent']['final_answer_prompt'])

    def run(self, query: str, context: str, user_preferences: str) -> str:
        prompt_value = self.prompt.format(query=query, context=context, user_preferences=user_preferences)
        return self.llm.invoke(prompt_value).content