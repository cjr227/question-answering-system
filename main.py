# Import Libraries
## Installed Libraries
import requests
import pandas as pd
from flask import Flask
from flask_restful import reqparse, Resource, Api

import torch
from torch import bfloat16
import transformers

from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()


# Get Data
url = "https://november7-730026606190.europe-west1.run.app/messages"
r = requests.get(url, params={"limit": 4000}) # Max number of records is 3349
df = pd.DataFrame(r.json()["items"])
assert len(df) == 3349

# Data storage (RAG)
## Loading dataframe content into a document
articles = DataFrameLoader(df, 
                           page_content_column = "message")

## Loading entire dataframe into document format
document = articles.load()                        

# Splitting document into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                chunk_overlap = 20)
splitted_texts = splitter.split_documents(document)

## Loading model to create the embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = InMemoryVectorStore(embedding_model)

# Index chunks
_ = vector_store.add_documents(documents=splitted_texts)

# Load LLM
# Configuring BitsAndBytesConfig for loading model in an optimal way
quantization_config = transformers.BitsAndBytesConfig(load_in_4bit = True,
                                        bnb_4bit_quant_type = 'nf4',
                                        bnb_4bit_use_double_quant = True,
                                        bnb_4bit_compute_dtype = bfloat16)
                                        
llm = HuggingFacePipeline.from_model_id(model_id = "HuggingFaceTB/SmolLM3-3B",
                                       task = 'text-generation',
                                       model_kwargs = {'quantization_config': quantization_config},
                                       device_map = "auto")

# RAG Chain
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=20)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        '''
        You are a personal assistant that answers questions about users. 
        Those users have asked you hundreds of questions in the past year, and their questions are saved in a database to which you have access.
        You have access to these messages from each user, and you can use their messages to answer new questions. 
        Answer the given question only using messages that are from the user mentioned in the question.
        Give your answer in one or two sentences.
        Only answer the input question. Do not answer anything else, ask any other questions or otherwise try to continue the conversation.

        Here are some examples of questions and answers:

        
        Question: What did John eat for breakfast on Friday?
        Answer: John ate green eggs and ham.
        

        Question: How many miles did Vanessa walk?
        Answer: Vanessa walked a thousand miles.
        

        Question: At what hotel did Jacob stay during his vacation?
        Answer: Jacob stayed at the Hotel California.
        
        
        Answer the given question using the following user messages in your response:
        '''
        f"\n\n{docs_content}"
    )

agent = create_agent(llm, tools=[], middleware=[prompt_with_context])


def answer_question(agent, query):
    results = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return {"answer": results["messages"][-1].text}


app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument("query", type=str)

class QuestionAnsweringSystem(Resource):
    """
    API endpoint for answering questions about members
    """
    def get(self, query):
        answer = answer_question(agent, query)
        return answer, 200

api.add_resource(QuestionAnsweringSystem, "/ask_question")

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
