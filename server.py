from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import openai
import requests

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# Import LangChain components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

# Define constants
CHROMA_PATH = "chroma"
# Updated prompt template to encourage a detailed, informative answer
PROMPT_TEMPLATE = """
Given the following context, provide a detailed and informative answer to the question.
Do not simply repeat the question; use the context to generate a unique and helpful response.

Context:
{context}

Question:
{question}

Answer:
"""

# Initialize Flask app
app = Flask(__name__)

# Preload the vector store and models at startup for faster queries
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
model = ChatOpenAI()

# URL for the TTS server (assumed to be running on port 5000 with endpoint /tts)
TTS_URL = "http://localhost:5000/tts"

@app.route('/query', methods=['GET', 'POST'])
def query():
    # Get the query text from URL params (GET) or JSON payload (POST)
    query_text = request.args.get('query')
    if not query_text:
        data = request.get_json(silent=True)
        if data and 'query' in data:
            query_text = data['query']
    if not query_text:
        return jsonify({"error": "No query provided. Please include a 'query' parameter."}), 400

    # Retrieve the top 3 most relevant document chunks using similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return jsonify({"error": "Unable to find matching results."}), 404

    # Combine document chunks into a context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Debug: Print the prompt
    print("Prompt sent to model:", prompt)
    
    # Generate the response using the ChatOpenAI model
    response_text = model.predict(prompt)
    
    # Debug: Print the generated AI response
    print("AI Response generated:", response_text)
    
    # Collect source metadata from the retrieved documents (if available)
    sources = [doc.metadata.get("source", None) for doc, _ in results]

    # Send the AI-generated response to the TTS server
    try:
        tts_payload = {"text": response_text}
        tts_response = requests.post(TTS_URL, json=tts_payload)
        if tts_response.ok:
            tts_result = tts_response.json()
        else:
            tts_result = {"error": tts_response.text}
    except Exception as e:
        tts_result = {"error": str(e)}

    return jsonify({
        "response": response_text,
        "sources": sources,
        "tts_result": tts_result
    })

if __name__ == '__main__':
    # Run the server on port 3000 and listen on all interfaces
    app.run(host='0.0.0.0', port=3000)
