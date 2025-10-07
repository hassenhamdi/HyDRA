import yaml
import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from .memory_agent import HydraMemoryAgent
import warnings
warnings.filterwarnings("ignore")

class PostInteractionAnalyzer:
    """
    Analyzes a completed conversation transcript to extract user preferences
    and provide feedback for the HELP/SIMPSON learning loop.
    """
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0.0)
        self.memory_agent = HydraMemoryAgent()
        with open("configs/agents.yaml", 'r') as f:
            prompts = yaml.safe_load(f)['post_interaction_analyzer']
            self.preference_prompt = PromptTemplate.from_template(prompts['preference_inference_prompt'])

    def analyze_and_learn(self, transcript: str, user_id: str, session_id: str):
        """
        Analyzes the transcript to infer preferences and saves them to memory.
        """
        try:
            # 1. Infer User Preferences
            prompt = self.preference_prompt.format(transcript=transcript)
            response = self.llm.invoke(prompt).content
            
            # Clean the response to ensure it's valid JSON
            clean_response = response.strip().replace("```json", "").replace("```", "")
            
            try:
                inferred_preferences = json.loads(clean_response)
                if isinstance(inferred_preferences, list):
                    for pref in inferred_preferences:
                        print(f"Inferred preference for {user_id}: {pref}")
                        self.memory_agent.save_preference(user_id, session_id, pref)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode preferences JSON: {clean_response}")

            # Note: The strategic feedback (policy memory) is already saved
            # by the AdaptiveCoordinator during the reasoning loop. This agent's
            # primary role is the post-hoc analysis of the full conversation.

        except Exception as e:
            print(f"Error during post-interaction analysis: {e}")
