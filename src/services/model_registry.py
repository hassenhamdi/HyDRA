# src/services/model_registry.py
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from src.utils.config_loader import get_config

class ModelRegistry:
    """
    A centralized registry to ensure that large models (embedding, reranker)
    are loaded into memory only once.
    """
    _embedding_model: BGEM3EmbeddingFunction = None
    _reranker_model: BGERerankFunction = None

    @classmethod
    def initialize_models(cls):
        """
        Loads the embedding and reranker models based on the global config.
        This should be called once at application startup.
        """
        if cls._embedding_model is not None:
            # Already initialized
            return

        print("--- Initializing BGE Models (Embedding & Reranker)... ---")
        config = get_config()
        embedding_config = config.get('embedding', {})
        use_fp16 = embedding_config.get('use_fp16', False)
        device = "cuda" if use_fp16 else "cpu"

        try:
            cls._embedding_model = BGEM3EmbeddingFunction(
                use_fp16=use_fp16,
                device=device
            )
            cls._reranker_model = BGERerankFunction(device=device)
            print(f"--- Models successfully initialized on {'GPU (FP16)' if use_fp16 else 'CPU (FP32)'}. ---")
        except Exception as e:
            print(f"FATAL: Failed to initialize models: {e}")
            raise

    @classmethod
    def get_embedding_model(cls) -> BGEM3EmbeddingFunction:
        """Returns the singleton instance of the embedding model."""
        if cls._embedding_model is None:
            raise RuntimeError("ModelRegistry has not been initialized. Call initialize_models() first.")
        return cls._embedding_model

    @classmethod
    def get_reranker_model(cls) -> BGERerankFunction:
        """Returns the singleton instance of the reranker model."""
        if cls._reranker_model is None:
            raise RuntimeError("ModelRegistry has not been initialized. Call initialize_models() first.")
        return cls._reranker_model