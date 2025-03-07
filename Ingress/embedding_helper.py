import openai
import constants

OPENAI_KEY = constants.OPENAI_KEY

# Initialize OpenAI API key
openai.api_key = OPENAI_KEY

class EmbeddingModel:    
    def get_embedding(self, text, model=constants.EMBEDDING_MODEL):
        text = text.replace("\n", " ")
        response = openai.Embedding.create(input=text, model=model)
        return response['data'][0]['embedding']