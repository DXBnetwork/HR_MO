import asyncio
import uuid
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import Memory, StaticMemoryBlock
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.agent.workflow import ReActAgent
from outlooktool import make_search_tool, email_tool

load_dotenv()

app = Flask(__name__)

# Initialize once at startup
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index = pc.Index("hrrag-1")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
)

llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"]
)

# Session store: session_id -> (memory, agent, session_state)
sessions: dict = {}


def get_or_create_session(session_id: str):
    if session_id in sessions:
        return sessions[session_id]

    memory = Memory.from_defaults(
        session_id=session_id,
        token_limit=6000,
        chat_history_token_ratio=0.7,
        token_flush_size=500,
        memory_blocks=[
            StaticMemoryBlock(
                name="firm_context",
                static_content=(
                    "You are an HR assistant for Murray Osorio PLLC, an immigration law firm. "
                    "Answer questions about company policies, benefits, and procedures accurately. "
                    "If a question is outside HR policy documents, offer to email HR directly. "
                    "If the user asks to send an email to HR, call the send_email tool. "
                    "Always include citations from the search tool in your final answer."
                ),
                priority=0,
            )
        ],
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
    )

    session_state = {"citations": []}

    def save_citations(citations):
        session_state["citations"] = citations

    search_tool = make_search_tool(query_engine, save_citations=save_citations)

    agent = ReActAgent(
    tools=[search_tool, email_tool],
    llm=llm,
    verbose=True,
    )

    sessions[session_id] = (memory, agent, session_state)
    return memory, agent, session_state


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    session_id = data.get('session_id') or str(uuid.uuid4())

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    memory, agent, session_state = get_or_create_session(session_id)

    async def run_agent():
        session_state["citations"] = []
        response = await agent.run(user_msg=question, memory=memory)
        return str(response), session_state["citations"]

    try:
        answer, citations = asyncio.run(run_agent())
        return jsonify({'answer': answer, 'session_id': session_id, 'citations': citations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
