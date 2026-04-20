import os
import pandas as pd
from dotenv import load_dotenv
from typing import List, Tuple
from langchain_core.documents import Document
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
class IngestionPipeline:
    """
    Class to handle data transformation and ingestion into AstraDB vector store.
    """
    def __init__(self):
        """
        Initialize environment variables, embedding model, and set CSV file path.
        """
        print("Initializing DataIngestion pipeline...")
        self.model_loader=ModelLoader()
        self.config=load_config()
        self._load_environment_variables()
        self.csv_path = self._get_csv_file_path()
        self.product_data = self._load_csv_data()
        



    def _load_environment_variables(self):
        """
        Load environment variables from .env file.
        """
        load_dotenv()

        required_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
     

    def _get_csv_file_path(self) -> str:
        """
        Get the CSV file path from environment variables.
        """
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, 'data', 'flipkart_product_review.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
        
        return csv_path
    
    def _load_csv_data(self) -> pd.DataFrame:
        """
        Load data from the CSV file into a pandas DataFrame.
        
        """
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_title', 'rating', 'summary', 'review'}
        if not expected_columns.issubset(df.columns):
            raise ValueError(f"CSV file is missing required columns. Expected columns: {expected_columns}")
        
        return df
    
    def _transform_data(self) -> List[Document]:
        """
        Transform the DataFrame into a list of Document objects.
        """
        
        product_list = []
        for _, row in self.product_data.iterrows():
            product_entry = {
                "product_name": row['product_title'],
                "rating": row['rating'],
                "summary": row['summary'],
                "review": row['review']
            }
            product_list.append(product_entry)

        documents = []
        for product in product_list:
            metadata = {
                "product_name": product['product_name'],
                "rating": product['rating'],
                "summary": product['summary']
            }
            doc = Document(page_content=product['review'], metadata=metadata)
            documents.append(doc)
        print(f"Transformed {len(documents)} documents.")
        return documents

    def store_in_vector_db(self, documents: List[Document]):
        """
        Store documents into Pinecone vector store.
        """
        index_name=self.config["pinecone_db"]["PINECONE_INDEX_NAME"]
        
        embeddings = self.model_loader.load_embeddings()
        
        # Step 1: Make Pinecone client 
        pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Step 2: Index create if not exit 
        existing_indexes = [i.name for i in pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"Creating index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=384,        # snowflake-arctic-embed:22m = 384
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # free tier region
                )
            )
            print(f"✅ Index '{index_name}' created.")
        else:
            print(f"⚠️ Index '{index_name}' already exists, using it.")
        
        # Step 3: Documents store
        print("Storing documents in Pinecone...")
        vstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name,
        )
        
        print(f"✅ Successfully stored {len(documents)} documents in Pinecone.")
        return vstore, [str(i) for i in range(len(documents))]

    def run_pipeline(self):
        """
        Run the entire data ingestion pipeline: load, transform, and store data.
        """
        print("Running data ingestion pipeline...")
        documents = self._transform_data()
        vector_store, inserted_ids = self.store_in_vector_db(documents)
        
         # Optionally do a quick search
        query = "Can you tell me the low budget headphone?"
        results = vector_store.similarity_search(query)

        print(f"\nSample search results for query: '{query}'")
        for res in results:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")

# Run if this file is executed directly
if __name__ == "__main__":
    ingestion = IngestionPipeline()
    ingestion.run_pipeline()