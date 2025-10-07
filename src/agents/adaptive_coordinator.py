# src/agents/adaptive_coordinator.py
import os
import yaml
import uuid
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from .memory_agent import HydraMemoryAgent
from .executors.vector import AdvancedVectorSearchAgent
from .executors.deep_search import DeepSearchAgent
import warnings
import json
warnings.filterwarnings("ignore")

class AdaptiveCoordinator:
    def __init__(self, gemini_api_key: str, user_id: str, session_id: str):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0.0)
        self.user_id = user_id
        self.session_id = session_id
        self.memory_agent = HydraMemoryAgent()
        with open("configs/agents.yaml", 'r') as f:
            prompts = yaml.safe_load(f)['coordinator']
            self.delegation_prompt = PromptTemplate.from_template(prompts['delegation_prompt'])
            self.reflection_prompt = PromptTemplate.from_template(prompts['policy_reflection_prompt'])

        self.executors = {
            "AdvancedVectorSearchAgent": AdvancedVectorSearchAgent(),
            "DeepSearchAgent": DeepSearchAgent(),
        }
        self.executor_descriptions = "\n".join([f"- {name}: {agent.description}" for name, agent in self.executors.items()])

    def delegate_task(self, sub_task: str) -> tuple[str, str, str]:
        strategic_guidance = self.memory_agent.retrieve_strategic_guidance(self.user_id, sub_task)
        
        prompt = self.delegation_prompt.format(
            executor_descriptions=self.executor_descriptions,
            sub_task=sub_task,
            strategic_guidance=strategic_guidance
        )
        expert_name = self.llm.invoke(prompt).content.strip().replace("'", "").replace("\"", "")

        if expert_name in self.executors:
            response_dict = self.executors[expert_name].run(sub_task)
            
            result = response_dict['result']
            strategy = response_dict.get('strategy_used', 'N/A')
            action_trace = response_dict.get('action_trace', [])
            score = 1.0 if result and "not found" not in result.lower() else -0.5

            # --- Policy Reflection Step ---
            reflection_prompt_val = self.reflection_prompt.format(
                sub_task=sub_task,
                executor_name=expert_name,
                action_trace=json.dumps(action_trace, indent=2),
                result=result[:1000] # Truncate result for context window
            )
            inferred_policy = self.llm.invoke(reflection_prompt_val).content.strip()

            self.memory_agent.save_policy_feedback(
                self.user_id, self.session_id, sub_task, 
                expert_name, strategy, score, 
                action_trace, inferred_policy
            )
            return result, expert_name, strategy
        else:
            return f"Error: The coordinator selected an unknown executor '{expert_name}'. Valid executors are: {list(self.executors.keys())}", "Unknown", "N/A"
