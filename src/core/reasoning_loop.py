import re
import logging
from typing import Generator, Tuple
from src.agents.meta_planner import MetaPlannerAgent
from src.agents.adaptive_coordinator import AdaptiveCoordinator
from src.agents.synthesis import SynthesisAgent
from src.agents.post_interaction_analyzer import PostInteractionAnalyzer
from src.agents.memory_agent import HydraMemoryAgent
import warnings
warnings.filterwarnings("ignore")

BEGIN_SUBTASK_TOKEN = "<|begin_call_subtask|>"
END_SUBTASK_TOKEN = "<|end_call_subtask|>"
BEGIN_RESULT_TOKEN = "<|begin_subtask_result|>"
END_RESULT_TOKEN = "<|end_subtask_result|>"

logger = logging.getLogger(__name__)

class ReasoningLoop:
    def __init__(self, gemini_api_key: str, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.planner = MetaPlannerAgent(gemini_api_key)
        self.coordinator = AdaptiveCoordinator(gemini_api_key, user_id, session_id)
        self.synthesis_agent = SynthesisAgent(gemini_api_key)
        self.analyzer = PostInteractionAnalyzer(gemini_api_key)
        self.memory_agent = HydraMemoryAgent()

    def _extract_sub_task(self, text: str) -> str:
        match = re.search(f"{re.escape(BEGIN_SUBTASK_TOKEN)}(.*?){re.escape(END_SUBTASK_TOKEN)}", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_final_answer(self, text: str) -> str:
        match = re.search(r"\\boxed\{(.*)\}", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def run(self, query: str, callback) -> Generator[str, None, str]:
        """
        Main execution loop for the agent.
        Yields the final answer content chunk by chunk, and returns the report title.
        """
        max_loops = 10
        user_preferences = self.memory_agent.retrieve_preferences(self.user_id)
        prompt = self.planner.get_initial_prompt(query, user_preferences)
        full_transcript = f"User Query: {query}\n---\n"
        aggregated_context = []
        final_answer_complete = ""
        report_title = "untitled_report"

        for i in range(max_loops):
            callback(f"Loop {i+1}/{max_loops}: Meta-Planner is reasoning...", "Planning")
            planner_output_step = self.planner.generate_step(prompt)
            full_transcript += planner_output_step

            sub_task = self._extract_sub_task(planner_output_step)
            final_answer_boxed = self._extract_final_answer(planner_output_step)

            if final_answer_boxed or not sub_task:
                callback("Planner finished. Synthesizing final answer.", "Synthesis")
                final_context_str = "\\n\\n".join(aggregated_context)
                
                # Use the new synchronous synthesis agent
                report_title, final_answer_complete = self.synthesis_agent.run(query, final_context_str, user_preferences)
                
                # Yield the complete content to the TUI's live display
                yield final_answer_complete
                
                # Perform post-run analysis
                self.analyzer.analyze_and_learn(full_transcript, self.user_id, self.session_id)
                self.memory_agent.save_interaction_summary(self.user_id, self.session_id, query, final_answer_complete)
                return report_title

            if sub_task:
                callback(f"Executing sub-task: '{sub_task}'", "Coordination")
                result, expert, strategy = self.coordinator.delegate_task(sub_task)
                
                distilled_result = f"Result for sub-task '{sub_task}' (using {expert}/{strategy}):\n{result}"
                aggregated_context.append(distilled_result)
                result_block = f"\n{BEGIN_RESULT_TOKEN}\n{distilled_result}\n{END_RESULT_TOKEN}\n"
                
                prompt += planner_output_step + result_block
                full_transcript += result_block
                callback(f"Sub-task complete. Used {expert}.", "Execution")

        # Fallback synthesis if max loops are reached
        callback("Max loops reached. Synthesizing with available information.", "Synthesis")
        final_context_str = "\\n\\n".join(aggregated_context)
        report_title, final_answer_complete = self.synthesis_agent.run(query, final_context_str, user_preferences)
        yield final_answer_complete
        
        self.analyzer.analyze_and_learn(full_transcript, self.user_id, self.session_id)
        self.memory_agent.save_interaction_summary(self.user_id, self.session_id, query, final_answer_complete)
        return report_title