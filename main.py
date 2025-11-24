# Import Libraries
## Installed Libraries
import requests
import pandas as pd
from flask import Flask, jsonify
from flask_restful import reqparse, Resource, Api

import torch
from torch import bfloat16
import transformers

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from mem0 import Memory

from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()


# Custom Functions
def get_user_id(user_name: str, id_to_name: dict = name_to_id_lookup) -> str:
    """Retrieve user ID from user name"""
    return id_to_name[user_name]

def retrieve_context(query: str, user_id: str) -> list[dict]:
    """Retrieve relevant context from Mem0"""
    try:
        memories = memory.search(query, user_id=user_id)
        memory_list = memories['results']
        
        serialized_memories = ' '.join([mem["memory"] for mem in memory_list])
        context = [
            {
                "role": "system", 
                "content": f"Relevant information: {serialized_memories}"
            },
            {
                "role": "user",
                "content": query
            }
        ]
        return context
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        # Return empty context if there's an error
        return [{"role": "user", "content": query}]

def generate_response(user_input: str, context: list[dict]) -> str:
    """Generate a response using the language model"""
    chain = prompt | chat_model
    response = chain.invoke({
        "context": context,
        "input": user_input
    })
    return response.content

def save_interaction(user_id: str, user_input: str, assistant_response: str):
    """Save the interaction to Mem0"""
    try:
        interaction = [
            {
              "role": "user",
              "content": user_input
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]
        result = memory.add(interaction, user_id=user_id, infer=False)
        print(f"Memory saved successfully: {len(result.get('results', []))} memories added")
    except Exception as e:
        print(f"Error saving interaction: {e}")

def chat_turn(user_input: str, user_name: str) -> str:
    user_id = get_user_id(user_name)
    # Retrieve context
    context = retrieve_context(user_input, user_id)
    
    # Generate response
    response = generate_response(user_input, context)
    
    # Save interaction
    save_interaction(user_id, user_input, response)
    
    return response


# Custom Variables
embedding_model_id = "avsolatorio/GIST-all-MiniLM-L6-v2"
llm_model_id = "llamafactory/tiny-random-Llama-3"
reranker_model_id = 'cross-encoder/ms-marco-TinyBERT-L2-v2'

# Get Data
url = "https://november7-730026606190.europe-west1.run.app/messages"
r = requests.get(url, params={"limit": 4000}) # Max number of records is 3349
df = pd.DataFrame(r.json()["items"])
assert len(df) == 3349

# Initiate Vector Store
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_id)
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="mem0"  # Required collection name
)

# Load LLM
# Configuring BitsAndBytesConfig for loading model in an optimal way
quantization_config = transformers.BitsAndBytesConfig(load_in_4bit = True,
                                        bnb_4bit_quant_type = 'nf4',
                                        bnb_4bit_use_double_quant = True,
                                        bnb_4bit_compute_dtype = bfloat16)
   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
llm = HuggingFacePipeline.from_model_id(model_id=llm_model_id,
                                       task = 'text-generation',
                                        pipeline_kwargs=dict(repetition_penalty=1.1, 
                                                             temperature=0.6, 
                                                             top_p=0.95, 
                                                             return_full_text=False,
                                                             max_new_tokens=16384
                                                             ),
                                       model_kwargs={'quantization_config': quantization_config},
                                       device_map = DEVICE)                                        
chat_model = ChatHuggingFace(llm=llm)


# Intialize Memory Layer
config = {
    "vector_store": {
        "provider": "langchain",
        "config": {
            "client": vector_store
        },
    },
    "llm": {
        "provider": "langchain",
        "config": {"model": chat_model},
    },
    "embedder": {
        "provider": "langchain",
        "config": {
            "model": embedding_model
        },
    },
    "reranker": {
        "provider": "sentence_transformer",
        "config": {
            "model": reranker_model_id,
            "device": DEVICE,  # Use GPU if available
            "batch_size": 64 if DEVICE == "cuda" else 32  # high batch size for high memory GPUs
        }
    },
}

memory = Memory.from_config(config)

for idx, row in df.iterrows():
    memory.add(row["message"], user_id=row["user_id"], infer=False)

# Initiate Chatbot
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content='''
        You are a personal assistant that answers questions about users. 
        Those users have asked you hundreds of questions in the past year, and their questions are saved in a database to which you have access.
        You have access to these messages from each user, and you can use their messages to answer new questions. 
        Answer the given question only using messages that are from the user mentioned in the question.
        Give your answer in one or two sentences.
        Only answer the input question. Do not answer anyxthing else, ask any other questions or otherwise try to continue the conversation.

        Here are some examples of questions and answers:

        
        Question: What did John eat for breakfast on Friday?
        Answer: John ate green eggs and ham.
        

        Question: How many miles did Vanessa walk?
        Answer: Vanessa walked a thousand miles.
        

        Question: At what hotel did Jacob stay during his vacation?
        Answer: Jacob stayed at the Hotel California.
        
        
        Answer the given question using the appropriate user messages in your response.
        '''),
    MessagesPlaceholder(variable_name="context"),
    HumanMessage(content="{input}")
])

df_name_to_id = df.groupby(["user_name", "user_id"], dropna=False).size().reset_index()
name_to_id_lookup = {row["user_name"]: row["user_id"] for idx, row in df_name_to_id.iterrows()}




app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument("user_input", type=str)
parser.add_argument("user_name", type=str)

class QuestionAnsweringSystem(Resource):
    """
    API endpoint for answering questions about members
    """
    def get(self, user_input, user_name):
        try:
            answer = chat_turn(user_input, user_name)
            return answer, 200
        except:
            return {
                    "Error": f"Issue answering query '{user_input}' for user '{user_name}'"
                }, 400

    def handle_error(self, e):
        return jsonify({"error": str(e)}), 500        

api.add_resource(QuestionAnsweringSystem, "/ask_question")

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
