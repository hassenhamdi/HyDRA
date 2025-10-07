# src/retrieval/engine.py
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
from src.utils.config_loader import get_config
from src.services.model_registry import ModelRegistry
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction


load_dotenv()

class HyDRARetriever(BaseRetriever):
    top_k_initial: int = 20
    top_k_final: int = 5
    bge_m3_ef : BGEM3EmbeddingFunction = None
    reranker : BGERerankFunction = None
    milvus_client : MilvusClient = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get pre-loaded models from the central registry
        self.bge_m3_ef = ModelRegistry.get_embedding_model()
        self.reranker = ModelRegistry.get_reranker_model()
        
        # Initialize the Milvus client
        self.milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        config = get_config()
        milvus_config = config['milvus']
        collection_name = milvus_config['collection_name']
        search_params = milvus_config['search_params']

        query_embeddings = self.bge_m3_ef([query])
        dense_req = AnnSearchRequest(data=[query_embeddings['dense'][0]], anns_field="dense_vector", param=search_params, limit=self.top_k_initial)
        
        # Handle sparse embeddings, which may not always be present
        sparse_emb_dict = {}
        if 'sparse' in query_embeddings and hasattr(query_embeddings['sparse'][0], 'col'):
             sparse_emb_dict = dict(zip(query_embeddings['sparse'][0].col, query_embeddings['sparse'][0].data))
        
        sparse_req = AnnSearchRequest(data=[sparse_emb_dict], anns_field="sparse_vector", param={"metric_type": "IP"}, limit=self.top_k_initial)

        initial_results = self.milvus_client.hybrid_search(
            collection_name=collection_name, reqs=[sparse_req, dense_req],
            ranker=RRFRanker(), limit=self.top_k_initial, output_fields=["chunk_text", "source"]
        )
        
        if not initial_results or not initial_results[0]:
            return []

        candidate_docs_text = [res.entity.get("chunk_text", "") for res in initial_results[0]]
        reranked_results = self.reranker(query=query, documents=candidate_docs_text, top_k=self.top_k_final)

        final_documents = []
        for res in reranked_results:
            original_doc_info = initial_results[0][res.index].entity
            doc = Document(
                page_content=res.text,
                metadata={"source": original_doc_info.get("source"), "relevance_score": res.score}
            )
            final_documents.append(doc)
            
        return final_documents
