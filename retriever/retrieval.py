import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from typing import List
from langchain_core.documents import Document
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
from dotenv import load_dotenv

class Retriever:
    
    def __init__(self):
        self.model_loader=ModelLoader()
        self.config=load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever = None
    
    def _load_env_variables(self):
         
        load_dotenv()
         
        required_vars = ["PINECONE_API_KEY"]  # not requried google api key becuase i use ollama model locally
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
    

    def load_retriever(self):
        if not self.vstore:
            index_name = self.config["pinecone_db"]["PINECONE_INDEX_NAME"]
            
            self.vstore = PineconeVectorStore(
                embedding= self.model_loader.load_embeddings(),
                index_name=index_name,
            )
        if not self.retriever:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            retriever = self.vstore.as_retriever(search_kwargs={"k": top_k})
            print("Retriever loaded successfully.")
            return retriever
   

    
    def call_retriever(self,query:str)-> List[Document]:
        retriever=self.load_retriever()
        output=retriever.invoke(query)
        return output
        
    
if __name__=='__main__':
    retriever_obj = Retriever()
    user_query = "Can you suggest good budget laptops?"
    results = retriever_obj.call_retriever(user_query)

    for idx, doc in enumerate(results, 1):
        print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")