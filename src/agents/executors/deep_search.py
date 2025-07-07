# src/agents/executors/deep_search.py
from duckduckgo_search import DDGS

class DeepSearchAgent:
    def __init__(self):
        self.description = "Best for complex research tasks requiring up-to-date information from the internet. Uses DuckDuckGo for privacy-first web search."
        self.search_tool = DDGS()

    def _format_results(self, results: list) -> str:
        if not results: return "No search results found."
        formatted = ""
        for i, res in enumerate(results, 1):
            formatted += f"Result {i}:\n  Title: {res.get('title')}\n  Snippet: {res.get('body')}\n  URL: {res.get('href')}\n\n"
        return formatted

    def run(self, query: str) -> dict:
        print(f"DeepSearchAgent searching for: '{query}'")
        try:
            search_results = self.search_tool.text(query, max_results=5)
            formatted_results = self._format_results(search_results)
        except Exception as e:
            formatted_results = f"An error occurred during web search: {e}"
            
        return {"result": formatted_results, "strategy_used": "web_search"}