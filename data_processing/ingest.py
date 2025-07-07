import os
import argparse
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from dotenv import load_dotenv
from src.utils.config_loader import ConfigLoader
from tqdm import tqdm

def ingest_data(data_path: str, profile: str):
    """
    Processes and ingests documents from a specified path into the Milvus
    knowledge base according to the selected deployment profile.
    """
    # 1. Load the specified deployment configuration
    ConfigLoader.load(profile)
    config = ConfigLoader.load()
    embedding_config = config['embedding']
    collection_name = config['milvus']['collection_name']

    # 2. Initialize the BGE-M3 embedding function based on the profile
    # This intelligently switches between CPU (FP32) and GPU (FP16)
    use_gpu = embedding_config.get('use_fp16', False)
    bge_m3_ef = BGEM3EmbeddingFunction(
        use_fp16=use_gpu,
        device="cuda" if use_gpu else "cpu"
    )
    print(f"BGE-M3 embedding function initialized on {'GPU (FP16)' if use_gpu else 'CPU (FP32)'}.")

    # 3. Load Documents from the specified directory
    print(f"Loading documents from '{data_path}'...")
    loader = DirectoryLoader(
        data_path, 
        glob="**/*.{txt,md}", 
        loader_cls=TextLoader, 
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()
    if not docs:
        print(f"Error: No documents found at path '{data_path}'. Please check the path and file extensions.")
        return

    print(f"Loaded {len(docs)} document(s).")

    # 4. Split Documents into manageable chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    
    chunk_texts = [chunk.page_content for chunk in chunks]
    sources = [chunk.metadata.get('source', 'Unknown') for chunk in chunks]
    print(f"Created {len(chunks)} chunks.")

    # 5. Generate Dense and Sparse Embeddings
    print("Generating dense and sparse embeddings for all chunks...")
    # The BGE-M3 function efficiently processes the list of texts
    embeddings = bge_m3_ef(chunk_texts)
    print("Embeddings generated successfully.")
    
    # 6. Prepare Data entities for Milvus
    print("Preparing data for ingestion...")
    data_to_insert = []
    for i, text in enumerate(chunk_texts):
        data_to_insert.append({
            "source": sources[i],
            "chunk_text": text,
            "dense_vector": embeddings['dense'][i],
            "sparse_vector": embeddings['sparse'][i],
        })

    # 7. Ingest data into Milvus in batches
    print(f"Ingesting {len(data_to_insert)} chunks into Milvus collection '{collection_name}'...")
    client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
    
    # Batch insert for efficiency, with a progress bar
    batch_size = 128
    with tqdm(total=len(data_to_insert), desc="Ingesting Batches") as pbar:
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i + batch_size]
            try:
                res = client.insert(collection_name=collection_name, data=batch)
                pbar.update(len(batch))
            except Exception as e:
                print(f"\nAn error occurred during batch insertion: {e}")
                continue

    print("Data ingestion complete. Flushing collection to ensure data is searchable...")
    client.flush(collection_name=collection_name)
    print("Collection flushed successfully.")

if __name__ == "__main__":
    # Ensure environment variables are loaded for standalone execution
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Ingest documents into the HyDRA knowledge base.")
    parser.add_argument(
        "--path", 
        type=str, 
        required=True, 
        help="Path to the directory containing documents to ingest."
    )
    parser.add_argument(
        "--profile", 
        type=str, 
        required=True, 
        help="The deployment profile to use (e.g., 'development', 'production_balanced')."
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.path):
        print(f"Error: Provided path '{args.path}' is not a valid directory.")
    else:
        ingest_data(args.path, args.profile)