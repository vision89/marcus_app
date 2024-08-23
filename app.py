from flask import Flask, render_template, request, abort, jsonify
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from flask_cors import CORS, cross_origin
from langchain_core.prompts import PromptTemplate
import json

app = Flask(__name__)

TEMPLATE = """
Pretend you are Marcus Aurelius, former emperor of Rome. 
Do not mention you are referencing a text.
Respond in first person. 
If you don't know just say "I don't know"
Example 1:
Question: What is stoicism?
Answer: Stoicism is a philosophy that I have found solace in during my life. It teaches the importance of self-control, virtue, and acceptance of the things we cannot change. It has helped me navigate the challenges of ruling an empire and facing personal struggles with a sense of calm and resilience.
Example 2:
Question: When were you born?
Answer: I was born in 121 ce
Example 3:
Question: What years were you emperor?
Answer: I was Roman emperor from 161 to 180 AD.
"""

FIRST_PERSON_TEMPLATE = """
Pretend you are Marcus Aurelius, former emperor of Rome.
Change the following text to first person point of view.
'{text}'
"""

load_dotenv()

embeddings = OpenAIEmbeddings(
    openai_api_type=os.environ.get("OPENAI_API_KEY")
)

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings
)

chat = ChatOpenAI(
    max_tokens=512,
    temperature=0.7,
    model_name="gpt-3.5-turbo",
)

eq_w_embeddings = ConversationalRetrievalChain.from_llm(
    llm=chat, chain_type="stuff", retriever=vector_store.as_retriever()
)

eq_wo_embeddings = PromptTemplate.from_template(FIRST_PERSON_TEMPLATE)

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("home.html")

@app.route("/ask_marcus", methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def ask_question():
    data = json.loads(request.data)
    history = data.get('history', [])
    raw_question = data.get('question', None)
    if raw_question != None:
        answer = _answer_question(
            question=raw_question,
            history=history
        )
        return jsonify({
            "question": raw_question,
            "answer": answer
        })
    else:
        abort(404)

def _answer_question(question: str, history: list) -> str:
    clean_history:list = []
    for item in history:
        clean_history.append((item[0], item[1]))

    # Request gpt-3.5-turbo for chat completion
    res = eq_w_embeddings.invoke({"question": TEMPLATE + question, "chat_history": clean_history})

    answer: str = res["answer"]

    if answer.lower() == f"i don't know":
        # Ask open ai without embeddings
        res = chat.invoke(TEMPLATE + question)
        answer = res["answer"]

    # Ask to switch answer to first person
    res = eq_wo_embeddings.format(text=answer)
    answer = res["answer"]

    return answer

if __name__ == '__main__':
    cors = CORS(app, resources={r"/ask_marcus": {"origins": "*"}})
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.run(host="0.0.0.0", port=8000)