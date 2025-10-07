# src/agents/executors/vector.py
import os
import yaml
import json
import re
import asyncio
import nest_asyncio
import logging
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retrieval.engine import HyDRARetriever
import warnings
warnings.filterwarnings("ignore")

nest_asyncio.apply()
logger = logging.getLogger(__name__)

class AdvancedVectorSearchAgent:
    def __init__(self):
        self.description = "Best for semantic or conceptual questions. Iteratively queries the internal knowledge base, reflects on the results, and refines its approach to find the best possible answer."
        self.retriever = None
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        
        with open("configs/agents.yaml", 'r') as f:
            config = yaml.safe_load(f)['advanced_vector_search_agent']
        
        self.hyde_prompt = PromptTemplate.from_template(config['hyde_generation_prompt'])
        self.think_prompt = PromptTemplate.from_template(config['iterative_think_prompt'])
        self.synthesis_prompt = PromptTemplate.from_template(config['iterative_synthesis_prompt'])

    def _get_retriever(self):
        if not self.retriever:
            self.retriever = HyDRARetriever()
        return self.retriever

    async def _think(self, query: str, history: str, iterations: int) -> dict:
        """Decide the next action to take."""
        prompt = self.think_prompt.format(query=query, history=history, iterations=iterations)
        response = await self.llm.ainvoke(prompt)
        try:
            clean_response = re.sub(r'```json\n|```', '', response.content).strip()
            return json.loads(clean_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse thought JSON: {response.content}")
            return {"thought": "Failed to parse my own thoughts. I will finish.", "action": "FINISH"}

    def _execute_retrieval(self, search_query: str) -> str:
        """Performs a single retrieval cycle and returns a formatted string."""
        retriever = self._get_retriever()
        docs = retriever.invoke(search_query)
        
        if not docs:
            return "No relevant information found in the knowledge base for this query."
            
        # Return a condensed version for the history
        return "\n".join([f"- Source: {doc.metadata.get('source', 'N/A')}, Snippet: {doc.page_content[:250]}..." for doc in docs])

    async def _run_async(self, query: str) -> dict:
        history = ""
        action_trace = []
        max_iterations = 5

        for i in range(max_iterations):
            decision = await self._think(query, history, i)
            thought = decision.get("thought")
            action = decision.get("action", "FINISH").upper()
            args = decision.get("args", {})
            
            history += f"\n--- Loop {i+1}: Thought ---\n{thought}\n"

            if action == "FINISH":
                break

            search_query = args.get("query")
            if not search_query:
                history += "Observation: Invalid action. Query argument was missing.\n"
                continue
            
            history += f"Action: {action} with query '{search_query}'\n"
            action_trace.append({"action": action, "args": {"query": search_query}})
            
            if action == "QUERY":
                observation = self._execute_retrieval(search_query)
                history += f"Observation:\n{observation}\n"

            elif action == "HYDE_QUERY":
                hyde_doc_prompt = self.hyde_prompt.format(query=search_query)
                hypothetical_doc = await self.llm.ainvoke(hyde_doc_prompt)
                observation = self._execute_retrieval(hypothetical_doc.content)
                history += f"Observation (from HyDE doc):\n{observation}\n"
        
        # Final Synthesis Step
        synthesis_prompt_val = self.synthesis_prompt.format(query=query, history=history)
        final_report = await self.llm.ainvoke(synthesis_prompt_val)
        
        return {
            "result": final_report.content, 
            "strategy_used": "iterative_vector_search",
            "action_trace": action_trace
        }

    def run(self, query: str) -> dict:
        """Synchronous wrapper for the async run method."""
        return asyncio.run(self._run_async(query))