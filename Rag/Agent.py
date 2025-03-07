import openai
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from embedding_helper import EmbeddingModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import constants
import prompts
from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel

class Agent:
    def __init__(self):
        self.app = FastAPI()
        self._setup_routes()

        openai.api_key = constants.OPENAI_KEY

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=constants.QDRANT_URL,
            api_key=constants.QDRANT_API_KEY,
            timeout=12
        )

        # Define your embedding model
        self.embedding_model = EmbeddingModel()

        # Initialize the FinBERT model and tokenizer
        self.finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

        
    def generate_json_from_user_input(self, user_input):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompts.instructions_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=1024,  # Adjust max tokens according to expected output size
                temperature=0  # Low temperature for more deterministic results
            )
            
            json_text = response.choices[0].message['content'].strip().replace("\n", "").replace("`", "").replace("json","")
            json_object = json.loads(json_text)   
            return json_object
        except json.JSONDecodeError:
            return "Failed to convert the AI result into a JSON object. The response might not be a valid JSON."
        except Exception as e:
            return f"Failed to generate JSON: {e}"


    def rerank_results_with_finbert(self, search_results, user_query):
        reranked_results = []

        for result in search_results:
            text = result.payload['text']
            
            # Prepare input for FinBERT
            inputs = self.finbert_tokenizer(user_query, text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            
            # Perform inference with FinBERT
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                scores = outputs.logits.squeeze()
                relevance_score = scores[1].item()  # Assuming index 1 is the relevance score for binary classification

            reranked_results.append((result, relevance_score))
        
        # Sort results by relevance score in descending order
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the reranked search results, ignoring the scores
        return [result[0] for result in reranked_results][:10]


    def search_qdrant(self, json_filters, user_input, enable_reranking):
        try:
            user_embedding = self.embedding_model.get_embedding(user_input)
            
            combined_conditions = []       
            for json_filter in json_filters:
                condition = Filter(
                    must=[
                        FieldCondition(key="ticker", match=MatchValue(value=json_filter["ticker"])),
                        FieldCondition(key="year", match=MatchValue(value=int(json_filter["year"]))),
                        FieldCondition(key="quarter", match=MatchValue(value=json_filter["quarter"])),
                        FieldCondition(key="report_type", match=MatchValue(value=json_filter["report_type"]))
                    ]
                )
                combined_conditions.append(condition)
            combined_filter = Filter(should=combined_conditions)
            
            search_result = self.qdrant_client.search(
                collection_name=constants.COLLECTION_NAME,
                query_vector=user_embedding,
                # limit=10 * len(json_filters),
                limit=50,
                query_filter=combined_filter
            )

            reranked_results = search_result
            if enable_reranking:
                # Rerank the search results using FinBERT
                reranked_results = self.rerank_results_with_finbert(search_result, user_input)
            
            return reranked_results
        except Exception as e:
            return f"Failed to search Qdrant: {e}"


    def generate_final_result_from_gpt(self, search_results, user_query):
        try:
            result_text = "Here are the top 10 most relevant financial data points found based on your query:\n"
            for idx, item in enumerate(search_results):
                result_text += f"Result {idx + 1}:\n{item.payload['text']}\n"

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are a financial analyst. The user has requested financial analysis or details. "
                        "You have been provided with search results from a financial database that match the user's query. "
                        "Please analyze the user's query using the data provided in the 'search results' section below and "
                        "summarize or answer the query using this information. Be sure to reference specific data points "
                        "from the search results to support your answer."
                    )},
                    {"role": "user", "content": f"User Query: {user_query}"},
                    {"role": "assistant", "content": f"Search Results:\n{result_text}"}
                ],
                max_tokens=4096,  # Adjust max tokens as needed for a detailed response
                temperature=0.2  # Use a low temperature for a more focused and factual response
            )

            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"Failed to generate the final result: {e}"


    def _setup_routes(self):
        @self.app.post("/search")
        def get_result(request: Request, body: SearchRequest):
            response = []
            try:
                json_result = self.generate_json_from_user_input(body.query)
                if isinstance(json_result, list):
                    search_results = self.search_qdrant(json_result, body.query, False)
                    
                    if isinstance(search_results, list):
                        if len(search_results) == 0:
                            return "No relevant financial data found."
                        final_result = self.generate_final_result_from_gpt(search_results, body.query)
                        response.append(f"{json_result}\nFinal Result from GPT:\n{final_result}" )

                    else:
                        response.append(search_results)
                else:
                    response.append(json_result)
                return {"answer": response}
            except Exception as e:
                return {"error": str(e)}, 500


    def run_api(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)


    def run_console_app(self, enable_reranking = True):
        print("Welcome to the Financial Analyst Helper Console App!")
        print("Type 'exit' to quit the application.\n")
        
        while True:
            query = input("Please enter your financial report query: ")
            
            if query.lower() == 'exit':
                print("Exiting the application. Goodbye!")
                break

            json_result = self.generate_json_from_user_input(query)
            print(json_result)

            if isinstance(json_result, list):
                search_results = self.search_qdrant(json_result, query, enable_reranking)
                
                if isinstance(search_results, list):
                    if len(search_results) == 0:
                        print("No relevant financial data found.")
                        break
                    final_result = self.generate_final_result_from_gpt(search_results, query)
                    print("\nFinal Result from GPT:\n", final_result)
                else:
                    print(search_results)
            else:
                print(json_result)

class SearchRequest(BaseModel):
    query: str

# Start the console application
if __name__ == "__main__":
    # Agent().run_console_app(False)
    Agent().run_api()