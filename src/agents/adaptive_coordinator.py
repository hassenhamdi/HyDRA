# src/agents/adaptive_coordinator.py
import os
import yaml
import uuid
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from .memory_agent import HydraMemoryAgent
from .executors.vector import AdvancedVectorSearchAgent
from .executors.deep_search import DeepSearchAgent

class AdaptiveCoordinator:
    def __init__(self, gemini_api_key: str, user_id: str, session_id: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=gemini_api_key, temperature=0.0)
        self.user_id = user_id
        self.session_id = session_id
        self.memory_agent = HydraMemoryAgent()
        with open("configs/agents.yaml", 'r') as f:
            self.prompt = PromptTemplate.from_template(yaml.safe_load(f)['coordinator']['delegation_prompt'])
        
        self.executors = {
            "AdvancedVectorSearchAgent": AdvancedVectorSearchAgent(),
            "DeepSearchAgent": DeepSearchAgent(),
        }
        self.executor_descriptions = "\n".join([f"- {name}: {agent.description}" for name, agent in self.executors.items()])

    def delegate_task(self, sub_task: str) -> tuple[str, str, str]:
        strategic_guidance = self.memory_agent.retrieve_strategic_guidance(self.user_id, sub_task)
        
        prompt = self.prompt.format(
            executor_descriptions=self.executor_descriptions,
            sub_task=sub_task,
            strategic_guidance=strategic_guidance
        )
        expert_name = self.llm.invoke(prompt).content.strip().replace("'", "").replace("\"", "")

        if expert_name in self.executors:
            response_dict = self.executors[expert_name].run(sub_task) 
            result = response_dict['result']
            strategy = response_dict.get('strategy_used', 'N/A')
            score = 1.0 if result and "not found" not in result.lower() else -0.5
            self.memory_agent.save_policy_feedback(self.user_id, self.session_id, sub_task, expert_name, strategy, score)
            return result, expert_name, strategy
        else:
            return f"Error: The coordinator selected an unknown executor '{expert_name}'. Valid executors are: {list(self.executors.keys())}", "Unknown", "N/A"