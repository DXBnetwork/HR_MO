import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "hrrag-1"

# Create index only if it does not exist.
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # bge-small-en-v1.5 output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created Pinecone index: {index_name}")
else:
    print(f"Using existing Pinecone index: {index_name}")

# Get the index
pinecone_index = pc.Index(index_name)

# Create Pinecone vector store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)


docstore = SimpleDocumentStore()
# Try to load existing docstore if it exists
try:
    docstore = SimpleDocumentStore.from_persist_dir("./docstore")
    print("Loaded existing docstore")
except FileNotFoundError:
    print("Creating new docstore")


# Load documents
reader = SimpleDirectoryReader(input_dir="/Users/Anthony/downloads/HR_folder")
documents = reader.load_data()

# Create ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,  # Connect to Pinecone
    docstore=docstore,
)

# Run pipeline (ingests into Pinecone)
nodes = pipeline.run(documents=documents)
docstore.persist("./docstore")

print(f"Successfully ingested {len(nodes)} nodes into Pinecone index '{index_name}'")
