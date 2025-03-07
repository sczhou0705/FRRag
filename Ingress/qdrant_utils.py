from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import constants

class QdrantUtils:
    def __init__(self, qdrant_url, qdrant_api_key, collection_name):
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=12
        )
        
        # Collection name for Qdrant
        self.collection_name = collection_name

    def list_all_filenames(self):
        try:
            all_files = []

            # Get the total number of points in the collection
            total_points = self.qdrant_client.count(self.collection_name).count

            # Fetch points in batches if necessary
            points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=total_points,  # Adjust this limit if the collection is very large
                with_payload=True
            )

            # Extract filenames from each point's payload
            for point in points[0]:  # Points are returned as a tuple (ScoredPoint[], next offset)
                if 'file_name' in point.payload:
                    all_files.append(point.payload['file_name'])
            
            # Remove duplicates and return the list of filenames
            return list(set(all_files))

        except Exception as e:
            return f"An error occurred while listing filenames: {e}"

    def delete_entries_with_ticker(self, ticker_value):
        try:
            # Construct the filter for deleting points with the given ticker
            delete_filter = Filter(
                must=[
                    FieldCondition(key="ticker", match=MatchValue(value=ticker_value))
                ]
            )
            
            # Perform the deletion operation
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter
            )

            print(f"All entries with ticker = '{ticker_value}' have been deleted.")
        
        except Exception as e:
            print(f"An error occurred while deleting entries: {e}")

# Example usage
if __name__ == "__main__":
    qdrant_url = constants.QDRANT_URL
    qdrant_api_key = constants.QDRANT_API_KEY
    collection_name = constants.COLLECTION_NAME
    
    qdrant_utils = QdrantUtils(qdrant_url, qdrant_api_key, collection_name)

    # List all filenames
    # filenames = qdrant_utils.list_all_filenames()
    # print("Filenames in collection:", filenames)

    # Delete entries with ticker = 'NVDIA'
    # qdrant_utils.delete_entries_with_ticker("AMD")
