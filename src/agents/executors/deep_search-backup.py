import os
import asyncio
import yaml
import logging
from asyncddgs import aDDGS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

class DeepSearchAgent:
    def __init__(self):
        self.description = "Best for complex research. Performs a multi-stage process: 1. Searches the web to find relevant URLs. 2. Fetches the full content of those pages. 3. Summarizes and synthesizes the information into a comprehensive report."

        # The search tool will be initialized within the async method

        # LLM for summarization
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1)

        with open("configs/agents.yaml", 'r') as f:
            prompts = yaml.safe_load(f)
            self.summarizer_prompt_template = PromptTemplate.from_template(prompts['web_summarizer_prompt'])

        # --- MCP Tool Integration ---
        # Load MCP server configurations
        with open("configs/mcp_servers.yaml", 'r') as f:
            mcp_configs = yaml.safe_load(f)

        self.mcp_client = MultiServerMCPClient(mcp_configs)

        self.fetch_tool = None

        # The MCP fetch tool is initialized lazily in the _run_async method.

    async def _initialize_mcp(self):
        """Initializes the MCP client and tools asynchronously."""
        if self.fetch_tool:
            return

        mcp_tools = await self.mcp_client.get_tools()
        for tool in mcp_tools:
            if tool.name == 'fetch':
                self.fetch_tool = tool
                break

        if not self.fetch_tool:
            logger.warning("MCP 'fetch' tool not found. The DeepSearchAgent will only be able to return search snippets.")

    async def _summarize_content(self, query: str, content: str) -> str:
        """Summarizes a single piece of web content using the LLM."""
        if not content or content.startswith("Error:"):
            return content

        prompt = self.summarizer_prompt_template.format(webpage_content=content, query=query)
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error during content summarization: {e}")
            return f"Error during summarization: {e}"

    async def _run_async(self, query: str) -> dict:
        await self._initialize_mcp() # Lazy initialize MCP tools
        """The core asynchronous logic for the deep search."""
        logger.info(f"DeepSearchAgent starting deep search for: '{query}'")
        try:
            # Stage 1: Initial Search to get top URLs
            async with aDDGS(verify=False) as ddgs:
                search_results = await ddgs.text(query, max_results=5)

            if not search_results:
                return {"result": "No initial search results found.", "strategy_used": "deep_search"}

            # If the fetch tool isn't available, return snippets as a fallback
            if not self.fetch_tool:
                snippets = "\\n".join([f"Title: {r['title']}\\nSnippet: {r['body']}" for r in search_results])
                return {"result": f"Could not fetch full content. Search snippets:\\n{snippets}", "strategy_used": "shallow_search_fallback"}

            # Stage 2: Fetch Raw Content from top URLs using the MCP 'fetch' tool
            fetch_tasks = [self.fetch_tool.ainvoke({'url': res['href']}) for res in search_results]
            contents = await asyncio.gather(*fetch_tasks)

            # Stage 3: Summarize each page's content in parallel
            summarize_tasks = [self._summarize_content(query, content) for content in contents]
            summaries = await asyncio.gather(*summarize_tasks)

            # Stage 4: Synthesize the final report from summaries
            final_report = ""
            for i, (res, summary) in enumerate(zip(search_results, summaries)):
                final_report += f"Source {i+1}: {res['title']}\\n"
                final_report += f"URL: {res['href']}\\n"
                final_report += f"Summary:\\n{summary}\\n---\\n"

            return {"result": final_report, "strategy_used": "deep_search_mcp"}

        except Exception as e:
            logger.error(f"An error occurred during deep web search: {e}")
            return {"result": f"An error occurred during deep web search: {e}", "strategy_used": "deep_search_error"}

    def run(self, query: str) -> dict:
        """Synchronous wrapper for the async run method."""
        try:
            return asyncio.run(self._run_async(query))
        except RuntimeError as e:
            if "cannot run nested" in str(e):
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(self._run_async(query))
            else:
                raise
