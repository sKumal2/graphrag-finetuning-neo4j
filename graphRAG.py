import os
from langchain_chroma import Chroma
from langchain_graph_retriever import GraphRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#populating the vector store 
vectorstore = Chroma(
    collection_name="movies",  # Your collection
    embedding_function=embeddings,
    # Direct Chroma Cloud params – pulls from .env if not set
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database="graphRAG",  # Your database
)

#graph traversal retriever
movie_edges = [
    ("release_year", "release_year"), 
    ("movie_genre", "movie_genre"),
]
from langchain_graph_retriever import EagerStrategy
traversal_retriever = GraphRetriever(
    store=vectorstore,
    edges=movie_edges,
    strategy=EagerStrategy(),  # Using Strategy object instead of string
)

try:
    results = traversal_retriever.invoke("what movies were released in 1994?")
    
    if results:
        for doc in results:
            print(f"{doc.id}: {doc.page_content}")
    else:
        print("No results found")
        
except Exception as e:
    print(f"Error during retrieval: {e}")
