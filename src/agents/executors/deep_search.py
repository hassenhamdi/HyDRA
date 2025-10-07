import os
import asyncio
import yaml
import json
import logging
import re
from asyncddgs import aDDGS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
import nest_asyncio
import warnings
warnings.filterwarnings("ignore")

nest_asyncio.apply()
logger = logging.getLogger(__name__)

THINK_PROMPT = """
You are a specialized researcher. Your goal is to answer the following query: '{query}'

You have completed {iterations} of research. Here is the history of your previous actions and observations:
{history}

Based on this history, what is your next action? Your available actions are:
1.  **SEARCH**: If you need to find new information or URLs. Use a concise, targeted search query.
2.  **FETCH**: If you have identified a promising URL from a previous search and need to get its full content.
3.  **FINISH**: If you believe you have gathered enough information to provide a comprehensive answer.

Respond with a strict format of JSON object containing your thought process and the action to take, don't include any backticks enclosing the object start explicitly by curly bracket.

Example:
{{
  "thought": "The initial search gave me a few promising URLs. I will start by fetching the content of the most relevant one to see if it contains the answer.",
  "action": "FETCH",
  "args": {{"url": "https://example.com/article"}}
}}
"""

SYNTHESIZE_PROMPT = """
You are a research analyst. You have been provided with a history of research actions and the information gathered. Your task is to synthesize this information into a comprehensive, final report that answers the user's original query.

Original Query: '{query}'

Research History & Findings:
{history}

Please provide a final, well-structured report.
"""

class DeepSearchAgent:
    def __init__(self):
        self.description = "Best for complex research. Performs an iterative, multi-step process of searching, fetching, and synthesizing information to answer a query."
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        self.fetch_tool = None
        self.mcp_client = None

    async def _initialize_mcp(self):
        if self.mcp_client:
            return
        with open("configs/mcp_servers.yaml", 'r') as f:
            mcp_configs = yaml.safe_load(f)
        self.mcp_client = MultiServerMCPClient(mcp_configs)
        mcp_tools = await self.mcp_client.get_tools()
        for tool in mcp_tools:
            if tool.name == 'fetch':
                self.fetch_tool = tool
                break
        if not self.fetch_tool:
            logger.warning("MCP 'fetch' tool not found. Fetching will not be possible.")

    async def _think(self, query: str, history: str, iterations: int) -> dict:
        """Decide the next action to take."""
        prompt = THINK_PROMPT.format(query=query, history=history, iterations=iterations)
        response = await self.llm.ainvoke(prompt)
        try:
            # Clean the response to handle potential markdown code blocks
            clean_response = re.sub(r'```json\n|```', '', response.content).strip()
            return json.loads(clean_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse thought JSON: {response.content}")
            return {"thought": "Failed to parse my own thoughts. I will finish.", "action": "FINISH"}

    async def _run_async(self, query: str) -> dict:
        await self._initialize_mcp()
        history = ""
        action_trace = []
        max_iterations = 5

        for i in range(max_iterations):
            decision = await self._think(query, history, i)
            thought = decision.get("thought")
            action = decision.get("action", "FINISH").upper()
            args = decision.get("args", {})
            
            history += f"\n--- Loop {i+1}: Thought ---\n{thought}\n"

            if action == "SEARCH":
                search_query = args.get("query")
                action_trace.append({"action": "SEARCH", "args": args})
                if not search_query:
                    history += "Observation: Invalid search action. Query was missing.\n"
                    continue
                
                async with aDDGS(verify=False) as ddgs:
                    results = await ddgs.text(search_query, max_results=5)
                
                observation = "\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
                history += f"Action: SEARCH for '{search_query}'\nObservation:\n{observation}\n"

            elif action == "FETCH":
                url = args.get("url")
                action_trace.append({"action": "FETCH", "args": args})
                if not url:
                    history += "Observation: Invalid fetch action. URL was missing.\n"
                    continue
                if not self.fetch_tool:
                    history += "Observation: Cannot fetch, fetch tool is not available.\n"
                    continue

                content = await self.fetch_tool.ainvoke({'url': url, 'max_length':999999})
                
                # Summarize the fetched content
                summary_prompt = f"Original Query: {query}\n\nContent from {url}:\n{content}\n\nPlease provide a concise summary of the key information relevant to the original query."
                summary = await self.llm.ainvoke(summary_prompt)
                
                history += f"Action: FETCH URL '{url}'\nObservation:\n{summary.content}\n"

            elif action == "FINISH":
                break

        # Final Synthesis Step
        synthesis_prompt = SYNTHESIZE_PROMPT.format(query=query, history=history)
        final_report = await self.llm.ainvoke(synthesis_prompt)
        
        return {
            "result": final_report.content, 
            "strategy_used": "deep_search_react",
            "action_trace": action_trace
        }

    def run(self, query: str) -> dict:
        return asyncio.run(self._run_async(query))
