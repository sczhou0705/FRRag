import json
from Agent import Agent 

def handler(event, context):
    try:
        query = event.get("query")
        
        # Validate the input
        if query is None:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'query' parameter in request body"})
            }
        
        # Return the same query back in the response
        result = Agent().get_result(context)
        response = {
            "statusCode": 200,
            "body": json.dumps({"response": result})
        }
        return response
    
    except json.JSONDecodeError:
        # Handle invalid JSON in the request body
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON in request body"})
        }