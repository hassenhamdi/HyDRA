# src/retrieval/engine.py
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
from src.utils.config_loader import get_config

load_dotenv()

class HyDRARetriever(BaseRetriever):
    top_k_initial: int = 20
    top_k_final: int = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_config()
        embedding_config = config['embedding']
        use_gpu = embedding_config['use_fp16']
        
        self.bge_m3_ef = BGEM3EmbeddingFunction(use_fp16=use_gpu, device="cuda" if use_gpu else "cpu")
        self.milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
        self.reranker = BGERerankFunction(device="cuda" if use_gpu else "cpu")

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        config = get_config()
        milvus_config = config['milvus']
        collection_name = milvus_config['collection_name']
        search_params = milvus_config['search_params']

        query_embeddings = self.bge_m3_ef([query])
        dense_req = AnnSearchRequest(data=[query_embeddings['dense'][0]], anns_field="dense_vector", param=search_params, limit=self.top_k_initial)
        sparse_req = AnnSearchRequest(data=[query_embeddings['sparse'][0]], anns_field="sparse_vector", param={"metric_type": "IP"}, limit=self.top_k_initial)

        initial_results = self.milvus_client.hybrid_search(
            collection_name=collection_name, reqs=[sparse_req, dense_req],
            rerank=RRFRanker(), limit=self.top_k_initial, output_fields=["chunk_text", "source"]
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