import os
import uuid
from typing import List, Dict
from pymilvus import MilvusClient
from src.utils.config_loader import get_config
from src.services.model_registry import ModelRegistry
import warnings
warnings.filterwarnings("ignore")

class HydraMemoryAgent:
    """
    Handles all interactions with the Milvus-based long-term memory store.
    """
    def __init__(self):
        self.milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"), token=os.getenv("MILVUS_TOKEN"))
        self.collection_name = "hydra_memory_store"
        # Get the pre-loaded embedding model from the central registry
        self.embedding_fn = ModelRegistry.get_embedding_model()
    
    def retrieve_strategic_guidance(self, user_id: str, sub_task: str) -> str:
        """Retrieves past strategic feedback relevant to the current sub-task."""
        try:
            query_embeddings = self.embedding_fn([sub_task])['dense'][0]

            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embeddings],
                limit=5,
                filter=f"user_id == '{user_id}' and memory_type == 'policy'",
                output_fields=["content"]
            )
            
            if not results or not results[0]:
                return "No specific guidance available."
                
            guidance = "\n".join([f"- {res['entity']['content']}" for res in results[0]])
            return f"Based on past performance:\n{guidance}"
        except Exception as e:
            print(f"Error retrieving strategic guidance: {e}")
            return "Could not retrieve strategic guidance due to an error."

    def save_policy_feedback(self, user_id: str, session_id: str, sub_task: str, executor: str, strategy: str, score: float, action_trace: list, inferred_policy: str):
        """Saves the analyzed feedback and inferred policy from an action."""
        try:
            feedback_id = str(uuid.uuid4())
            
            # The searchable content is now the concise, inferred policy.
            # We embed the policy itself so we can find similar strategic lessons in the future.
            embeddings = self.embedding_fn([inferred_policy])
            
            data = [{
                "id": feedback_id,
                "user_id": user_id,
                "session_id": session_id,
                "memory_type": "policy",
                "content": inferred_policy, # Store the distilled learning
                "vector": embeddings['dense'][0],
                "metadata": {
                    "sub_task": sub_task,
                    "executor": executor,
                    "strategy": strategy,
                    "score": score,
                    "action_trace": action_trace # Store the full trace for context
                }
            }]
            
            self.milvus_client.insert(collection_name=self.collection_name, data=data)
        except Exception as e:
            print(f"Error saving policy feedback: {e}")

    def retrieve_preferences(self, user_id: str) -> str:
        """Retrieves all stored user preferences."""
        try:
            results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter=f"user_id == '{user_id}' and memory_type == 'preference'",
                output_fields=["content"]
            )
            
            if not results:
                return "No preferences recorded."
            
            preferences = "\n".join([f"- {res['content']}" for res in results])
            return preferences
        except Exception as e:
            print(f"Error retrieving preferences: {e}")
            return "Could not retrieve preferences due to an error."

    def save_preference(self, user_id: str, session_id: str, preference: str):
        """Saves a single inferred user preference."""
        try:
            preference_id = str(uuid.uuid4())
            embeddings = self.embedding_fn([preference])
            
            data = [{
                "id": preference_id,
                "user_id": user_id,
                "session_id": session_id,
                "memory_type": "preference",
                "content": preference,
                "vector": embeddings['dense'][0],
                "metadata": {}
            }]
            
            self.milvus_client.insert(collection_name=self.collection_name, data=data)
        except Exception as e:
            print(f"Error saving preference: {e}")

    def save_interaction_summary(self, user_id: str, session_id: str, query: str, final_answer: str):
        """Saves a summary of the completed interaction."""
        try:
            summary_id = str(uuid.uuid4())
            content = f"User asked: '{query}'. Final Answer: '{final_answer}'"
            embeddings = self.embedding_fn([content])
            
            data = [{
                "id": summary_id,
                "user_id": user_id,
                "session_id": session_id,
                "memory_type": "interaction_summary",
                "content": content,
                "vector": embeddings['dense'][0],
                "metadata": {"query": query}
            }]
            
            self.milvus_client.insert(collection_name=self.collection_name, data=data)
        except Exception as e:
            print(f"Error saving interaction summary: {e}")
