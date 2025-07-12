# src/services/milvus_setup.py
import os
import argparse
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
from src.utils.config_loader import ConfigLoader

MEMORY_COLLECTION = "hydra_memory_store"

def setup_milvus(profile: str='production_balanced'):
    ConfigLoader.load(profile)
    config = ConfigLoader.load()
    milvus_config = config['milvus']
    knowledge_collection_name = milvus_config['collection_name']

    print("Connecting to Milvus...")
    client = MilvusClient(uri='http://localhost:19530')
    
    if client.has_collection(knowledge_collection_name):
        client.drop_collection(knowledge_collection_name)
    
    print(f"Creating knowledge collection: {knowledge_collection_name}")
    knowledge_schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    knowledge_schema.add_field("id", DataType.INT64, is_primary=True)
    knowledge_schema.add_field("source", DataType.VARCHAR, max_length=1024)
    knowledge_schema.add_field("chunk_text", DataType.VARCHAR, max_length=8192)
    knowledge_schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
    knowledge_schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    dense_index_config = milvus_config['dense_index']
    index_params= client.prepare_index_params()
    
    index_params.add_index(
        field_name="dense_vector",
        index_type=dense_index_config['index_type'],
        metric_type=dense_index_config['metric_type'],
        params=dense_index_config['build_params']
    )


    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    
    client.create_collection(
        collection_name=knowledge_collection_name,
        schema=knowledge_schema,
        index_params=index_params
    )
    print("Knowledge collection created successfully.")

    if not client.has_collection(MEMORY_COLLECTION):
        print(f"Creating memory collection: {MEMORY_COLLECTION}")
        memory_schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        memory_schema.add_field("id", DataType.VARCHAR, max_length=36, is_primary=True)
        memory_schema.add_field("user_id", DataType.VARCHAR, max_length=256, index_name="idx_user_id")
        memory_schema.add_field("session_id", DataType.VARCHAR, max_length=256)
        memory_schema.add_field("memory_type", DataType.VARCHAR, max_length=50)
        memory_schema.add_field("content", DataType.VARCHAR, max_length=8192)
        memory_schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
        memory_schema.add_field("metadata", DataType.JSON) 
        
        index_params_memory = client.prepare_index_params()
        index_params_memory.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="IP")
        
        client.create_collection(
            collection_name=MEMORY_COLLECTION,
            schema=memory_schema,
            index_params=index_params_memory
        )

        print("Memory collection created successfully.")
        
    else:
        print(f"Collection '{MEMORY_COLLECTION}' already exists.")

    print("Milvus setup complete.")

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Initialize Milvus for HyDRA.")
    parser.add_argument("--profile", type=str, required=True, help="Deployment profile to use for setup.")
    args = parser.parse_args()
    setup_milvus(args.profile)
