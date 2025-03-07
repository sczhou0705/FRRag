import os
import sys
import shutil
import uuid
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from langchain_text_splitters import CharacterTextSplitter
# from openai.embeddings_utils import get_embedding
from embedding_helper import EmbeddingModel
from dateutil.parser import parse
import constants
import extract_items
import download_filings

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=constants.QDRANT_URL,
    api_key=constants.QDRANT_API_KEY,
    timeout=12
)

def create_collection():
    # Define Qdrant collection
    collection_name = constants.COLLECTION_NAME
    # Check if the collection exists
    existing_collections = qdrant_client.get_collections().collections
    collection_exists = any(collection.name == collection_name for collection in existing_collections)
    
    if not collection_exists:
        # Create the collection with vector parameters if it doesn't exist
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists. Proceeding to insert data.")

def download_10_k(tickers):
    try:
        config = {
            "download_filings": {
                "start_year": 2023,
                "end_year": 2024,
                "quarters": [1,2,3,4],
                "filing_types": ["10-K"],
                "cik_tickers": tickers,
                "user_agent": "zhoushichao888@gmail.com",
                "raw_filings_folder": "RAW_FILINGS",
                "indices_folder": "INDICES",
                "filings_metadata_file": "FILINGS_METADATA.csv",
                "skip_present_indices": True
            },
            "extract_items": {
                "raw_filings_folder": "RAW_FILINGS",
                "extracted_filings_folder": "EXTRACTED_FILINGS",
                "filings_metadata_file": "FILINGS_METADATA.csv",
                "filing_types": ["10-K"],
                "include_signature": False,
                "items_to_extract": [],
                "remove_tables": False,
                "skip_extracted_filings": True
            }
        }

        download_filings.main(config)
        extract_items.main(config)
    except Exception as ex:
        print(str(ex))

def download_10_q(tickers):
    try:
        config = {
            "download_filings": {
                "start_year": 2023,
                "end_year": 2024,
                "quarters": [1,2,3,4],
                "filing_types": ["10-K"],
                "cik_tickers": tickers,
                "user_agent": "zhoushichao888@gmail.com",
                "raw_filings_folder": "RAW_FILINGS",
                "indices_folder": "INDICES",
                "filings_metadata_file": "FILINGS_METADATA.csv",
                "skip_present_indices": True
            },
            "extract_items": {
                "raw_filings_folder": "RAW_FILINGS",
                "extracted_filings_folder": "EXTRACTED_FILINGS",
                "filings_metadata_file": "FILINGS_METADATA.csv",
                "filing_types": ["10-K"],
                "include_signature": False,
                "items_to_extract": [],
                "remove_tables": False,
                "skip_extracted_filings": False
            }
        }

        download_filings.main(config)
        extract_items.main(config)
    except Exception as ex:
        print(str(ex))

def upload_10_k_to_qdrant(directory, out_dir):
    standard_10_k_items = [
            "item_1",
            "item_1A",
            "item_1B",
            "item_1C",
            "item_2",
            "item_3",
            "item_4",
            "item_5",
            "item_6",
            "item_7",
            "item_7A",
            "item_8",
            "item_9",
            "item_9A",
            "item_9B",
            "item_9C",
            "item_10",
            "item_11",
            "item_12",
            "item_13",
            "item_14",
            "item_15",
            "item_16"
        ]
    cik_ticker_mapping = load_cik_ticker_mapping()
    create_collection()
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=3000, chunk_overlap=300)
    embedding = EmbeddingModel()
    total_file = 0
    complete_without_error_count = 0

    for root_dir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root_dir, file)

            if not file.endswith(".json") and "10K" in file.name:            
                continue

            try:
                total_file += 1
                with open(file_path, "r", encoding="utf-8") as file:
                    no_error = True
                    data = json.load(file)
                    filename = os.path.basename(file_path)
                    ticker = cik_ticker_mapping[str(data["cik"])]
                    if ticker is None:
                        raise Exception(f"Unable to fild ticker. {file_path}") 
                    results = []
                    print(f"Extracting {ticker} {data['period_of_report']} 10-K...")
                    for item_name in standard_10_k_items:
                        parsed_date = parse(data["period_of_report"])
                        formatted_date = parsed_date.strftime('%Y-%m-%d')                        
                        record = {
                            "filename": filename,
                            "ticker": ticker,
                            "company_name": data["company"],
                            "conformed_period": formatted_date,
                            "report_type": "10-K",
                            "item_name": item_name,
                            "content": data[item_name],
                            "year": parsed_date.year
                        }
                        if not check_10_k_item_content(item_name, record["content"]):
                            print(f"Error when extracing {item_name} of {ticker} {data['period_of_report']} 10-K. Filename: {os.path.basename(file_path)}")
                            no_error = False
                            break
                        if len(record["content"]) != 0:
                            results.append(record)
                    if no_error:
                        for i in results:
                            chunks = text_splitter.split_text(i["content"])
                            for idx, chunk in enumerate(chunks):
                                print(f"Processing chunk {idx + 1}/{len(chunks)}...")
                                vector = embedding.get_embedding(chunk)  # Replace this with your embedding model
                                points = []
                                points.append(PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector=vector,
                                    payload={
                                        "file_id": f"{i['report_type']}_{ticker}_{i['conformed_period']}_{i['item_name']}_{idx}",
                                        "file_name": filename,
                                        "ticker": ticker,
                                        "company_name": i["company_name"],
                                        "conformed_period": i["conformed_period"],
                                        "report_type": i["report_type"],
                                        "item_name": i["item_name"],
                                        "text": chunk,
                                        "year": i["year"],
                                        "quarter": "Q4",
                                    }
                                ))
                                qdrant_client.upsert(
                                    collection_name=constants.COLLECTION_NAME,
                                    points=points
                                )
                        print(f"Saved {ticker} {data['period_of_report']} 10-K to Qdrant...")
                        complete_without_error_count += 1
                if no_error:
                    shutil.move(file_path, out_dir)
            except Exception as e:
                print(f"Error when processing {file_path}: {str(e)}")

    print(f"Processed {complete_without_error_count}/{total_file} without errors.")

def check_10_k_item_content(item_name, content):
    content_clean = content.lower().replace(" ", "").replace("\n", "").replace("\t", "").replace("\u2007", "").replace(".", "", 1).replace(":", "", 1).replace("-", "", 1)
    if item_name in ["item_1", "item_1A", "item_7", "item_8", "item_5", "item_9A", "item_10", "item_11", "item_12", "item_15"] and content == "":
        return False
    if content != "" and not content_clean.startswith(item_name.lower().replace("_", "")):
        return False
    return True

def load_cik_ticker_mapping():
    dir = rf"{os.path.dirname(os.path.abspath(__file__))}\company_tickers_exchange.json"
    mapping = {}
    with open(dir, "r", encoding="utf-8") as file:
        data = json.load(file)
        
        fields = data["fields"]
        cik_index = fields.index("cik")
        ticker_index = fields.index("ticker")
        
        for row in data["data"]:
            mapping[str(row[cik_index])] = row[ticker_index]
    
    return mapping
         
if __name__ == "__main__":
    # download_10_k(['AMD','NVDA'])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    annual_directory = os.path.join(current_dir, "datasets", "EXTRACTED_FILINGS", "10-K")
    annual_out_directory = os.path.join(current_dir, "datasets", "EXTRACTED_FILINGS", "10-K_Completed")
    
    upload_10_k_to_qdrant(annual_directory, annual_out_directory)