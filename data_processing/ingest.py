import os
import argparse
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from dotenv import load_dotenv
from src.utils.config_loader import ConfigLoader, get_config
from src.services.model_registry import ModelRegistry
from tqdm import tqdm

def ingest_data(data_path: str, profile: str):
    """
    Processes and ingests documents from a specified path (file or directory)
    into the Milvus knowledge base.
    """
    # ConfigLoader.load(profile) is called from the __main__ block
    config = get_config()
    collection_name = config['milvus']['collection_name']

    # Get the pre-initialized embedding model from the registry
    bge_m3_ef = ModelRegistry.get_embedding_model()
    use_fp16 = config['embedding'].get('use_fp16', False)
    print(f"Using pre-loaded BGE-M3 embedding function on {'GPU (FP16)' if use_fp16 else 'CPU (FP32)'}.")

    # --- Smart Loading: Handle both file and directory paths ---
    print(f"Loading documents from '{data_path}'...")
    if os.path.isfile(data_path):
        loader = TextLoader(data_path)
        docs = loader.load()
    elif os.path.isdir(data_path):
        loader = DirectoryLoader(
            data_path, 
            glob="**/*.md",
            loader_cls=TextLoader, 
            show_progress=True,
            use_multithreading=True
        )
        docs = loader.load()
    else:
        print(f"Error: Path '{data_path}' is not a valid file or directory.")
        return

    if not docs:
        print(f"Error: No documents found at path '{data_path}'.")
        return

    print(f"Loaded {len(docs)} document(s).")

    # --- Standard Ingestion Pipeline ---
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    
    chunk_texts = [chunk.page_content for chunk in chunks]
    sources = [chunk.metadata.get('source', 'Unknown') for chunk in chunks]
    print(f"Created {len(chunks)} chunks.")

    print("Generating dense and sparse embeddings for all chunks...")
    embeddings = bge_m3_ef(chunk_texts)
    print("Embeddings generated successfully.")
    
    print("Preparing data for ingestion...")
    data_to_insert = []
    for i, text in enumerate(chunk_texts):
        sparse_dict = {}
        if hasattr(embeddings['sparse'][i], 'col'):
            sparse_dict = dict(zip(embeddings['sparse'][i].col, embeddings['sparse'][i].data))
        data_to_insert.append({
            "source": sources[i],
            "chunk_text": text,
            "dense_vector": embeddings['dense'][i],
            "sparse_vector": sparse_dict,
        })

    print(f"Ingesting {len(data_to_insert)} chunks into Milvus collection '{collection_name}'...")
    client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
    
    batch_size = 128
    with tqdm(total=len(data_to_insert), desc="Ingesting Batches") as pbar:
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i + batch_size]
            try:
                client.insert(collection_name=collection_name, data=batch)
                pbar.update(len(batch))
            except Exception as e:
                print(f"\nAn error occurred during batch insertion: {e}")
                continue

    print("Data ingestion complete. Flushing collection to ensure data is searchable...")
    client.flush(collection_name=collection_name)
    print("Collection flushed successfully.")

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Ingest documents into the HyDRA knowledge base.")
    parser.add_argument(
        "--path", 
        type=str, 
        required=True, 
        help="Path to the file or directory of documents to ingest."
    )
    parser.add_argument(
        "--profile", 
        type=str, 
        required=True, 
        help="The deployment profile to use (e.g., 'development')."
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"Error: Provided path '{args.path}' does not exist.")
    else:
        # Load config and initialize models *before* calling the main function
        ConfigLoader.load(args.profile)
        ModelRegistry.initialize_models()
        ingest_data(args.path, args.profile)
