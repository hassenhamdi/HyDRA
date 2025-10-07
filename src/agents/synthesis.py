# src/agents/synthesis.py
import yaml
import json
from typing import Tuple
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
warnings.filterwarnings("ignore")

class SynthesisAgent:
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0.3)
        with open("configs/agents.yaml", 'r') as f:
            self.prompt = PromptTemplate.from_template(yaml.safe_load(f)['synthesis_agent']['final_answer_prompt'])

    def run(self, query: str, context: str, user_preferences: str) -> Tuple[str, str]:
        """
        Generates a final report by making a single call to the LLM,
        parsing the JSON response, and returning the title and content.
        """
        prompt_value = self.prompt.format(query=query, context=context, user_preferences=user_preferences)
        
        # Make a single, non-streaming call to get the full response
        full_response = self.llm.invoke(prompt_value)
        full_response_text = full_response.content

        try:
            # Clean the response to remove potential markdown formatting
            clean_json_str = full_response_text.strip().replace("```json", "").replace("```", "")
            response_json = json.loads(clean_json_str)
            
            report_title = response_json.get("report_title", "untitled_report")
            report_content = response_json.get("report_content", "No content was generated.")

            return report_title, report_content

        except json.JSONDecodeError:
            # If the LLM fails to return valid JSON, return the raw response as content
            # with a default error title. This makes the agent resilient.
            error_content = "Error: The Synthesis Agent failed to generate a valid structured report. Displaying raw output:\n\n" + full_response_text
            return "synthesis_agent_error", error_content