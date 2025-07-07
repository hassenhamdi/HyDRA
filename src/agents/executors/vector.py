# src/agents/executors/vector.py
import os
import yaml
import json
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retrieval.engine import HyDRARetriever

class AdvancedVectorSearchAgent:
    def __init__(self):
        self.description = "Best for semantic or conceptual questions about the internal knowledge base. Can autonomously choose the best strategy (direct vs. hypothetical document) to find information."
        self.retriever = HyDRARetriever()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.0)
        
        with open("configs/agents.yaml", 'r') as f:
            config = yaml.safe_load(f)['advanced_vector_search_agent']
        
        self.strategy_prompt = PromptTemplate.from_template(config['strategy_selection_prompt'])
        self.hyde_prompt = PromptTemplate.from_template(config['hyde_generation_prompt'])

    def _decide_strategy(self, query: str) -> str:
        prompt = self.strategy_prompt.format(query=query)
        response = self.llm.invoke(prompt).content.lower().strip()
        return 'hyde' if 'hyde' in response else 'direct'

    def _generate_hyde_doc(self, query: str) -> str:
        prompt = self.hyde_prompt.format(query=query)
        return self.llm.invoke(prompt).content

    def run(self, query: str) -> dict:
        strategy = self._decide_strategy(query)
        
        if strategy == 'hyde':
            search_query = self._generate_hyde_doc(query)
        else:
            search_query = query
            
        docs = self.retriever.invoke(search_query)
        
        result_text = "\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in docs]) if docs else "No relevant information found in the knowledge base."
        
        return {"result": result_text, "strategy_used": strategy}