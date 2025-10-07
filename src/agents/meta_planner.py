import yaml
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
warnings.filterwarnings("ignore")

class MetaPlannerAgent:
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0.0)
        with open("configs/agents.yaml", 'r') as f:
            self.prompt_template = PromptTemplate.from_template(yaml.safe_load(f)['meta_planner']['planning_prompt'])

    def get_initial_prompt(self, query: str, user_preferences: str) -> str:
        """Creates the initial prompt to start the reasoning process."""
        return self.prompt_template.format(query=query, user_preferences=user_preferences)

    def generate_step(self, current_prompt: str) -> str:
        """
        Generates the next step in the reasoning chain.
        The response may contain a sub-task call or the final answer.
        """
        # The full reasoning history is part of the prompt
        response = self.llm.invoke(current_prompt)
        content = response.content
        # Handle cases where the model might return a list of content parts
        if isinstance(content, list):
            return "".join(str(part) for part in content)
        return str(content)
