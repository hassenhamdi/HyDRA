# src/core/reasoning_loop.py
from src.agents.meta_planner import MetaPlannerAgent
from src.agents.adaptive_coordinator import AdaptiveCoordinator
from src.agents.synthesis import SynthesisAgent
from src.agents.post_interaction_analyzer import PostInteractionAnalyzer
from src.agents.memory_agent import HydraMemoryAgent

class ReasoningLoop:
    def __init__(self, gemini_api_key: str, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.planner = MetaPlannerAgent(gemini_api_key)
        self.coordinator = AdaptiveCoordinator(gemini_api_key, user_id, session_id)
        self.synthesis_agent = SynthesisAgent(gemini_api_key)
        self.analyzer = PostInteractionAnalyzer(gemini_api_key)
        self.memory_agent = HydraMemoryAgent()

    def run(self, query: str, callback) -> str:
        callback("Generating strategic plan...", "Planning")
        plan = self.planner.generate_plan(query)
        callback(f"Plan created: {plan}", "Planning")

        execution_context = ""
        full_transcript = f"User Query: {query}\nPlan: {plan}\n---\n"
        user_preferences = self.memory_agent.retrieve_preferences(self.user_id)

        for sub_task in plan:
            callback(f"Executing sub-task: '{sub_task}'", "Coordination")
            result, expert, strategy = self.coordinator.delegate_task(sub_task)
            
            summary = f"Result for sub-task '{sub_task}' (using {expert}/{strategy}):\n{result}\n---\n"
            execution_context += summary
            full_transcript += summary
            callback(f"Sub-task complete. Used {expert}.", "Execution")
        
        callback("Synthesizing final answer...", "Synthesis")
        final_answer = self.synthesis_agent.run(query, execution_context, user_preferences)
        full_transcript += f"\nFinal Answer: {final_answer}"

        callback("Analyzing session for self-improvement...", "Learning")
        self.analyzer.analyze_and_learn(full_transcript, self.user_id, self.session_id)
        self.memory_agent.save_interaction_summary(self.user_id, self.session_id, query, final_answer)

        return final_answer