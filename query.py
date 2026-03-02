import os
import uuid
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import Memory, StaticMemoryBlock
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from braintrust import init_logger
from langchain_groq import ChatGroq
from llama_index.core.agent.workflow import ReActAgent
from outlooktool import make_search_tool, email_tool
load_dotenv()

try:
    from ragas.metrics import Faithfulness, AnswerRelevancy
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

logger = init_logger(project="My Project")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index = pc.Index("hrrag-1")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

llm = Groq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])
ragas_llm = ChatGroq(
    model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"]
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
                    "If a question is outside the scope of HR policies, offer to email HR directly. "
                    "If the user asks to send an email to HR, you must call the send_email tool immediately using the drafted content. "
                    "Do not write out the email as a response, call the tool. "
                    "Always include citations from the search tool in your final answer. "
                    "The HR email address is hr@murrayosorio.com."
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


def evaluate_with_ragas(question: str, answer: str, contexts: list[str]) -> tuple[float, float]:
    if not RAGAS_AVAILABLE:
        print("RAGAS is not installed. Install it with: pip install ragas")
        return 0.5, 0.5

    try:
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        dataset = EvaluationDataset(samples=[sample])
        results = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy()],
            llm=ragas_llm,
        )
        row = results.to_pandas().iloc[0]
        faithfulness = float(row.get("faithfulness", 0.5))
        relevancy = float(row.get("answer_relevancy", 0.5))
        return max(0.0, min(1.0, faithfulness)), max(0.0, min(1.0, relevancy))
    except Exception as exc:
        print(f"RAGAS evaluation failed: {exc}")
        return 0.5, 0.5


def ask_question(query: str, session_id: str = None):
    # Generate a session ID if one isn't provided
    if session_id is None:
        session_id = str(uuid.uuid4())
        print(f"New session started: {session_id}")

    memory, agent, session_state = get_or_create_session(session_id)

    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Question: {query}")
    print(f"{'='*60}\n")

    async def run_agent():
        session_state["citations"] = []
        response = await agent.run(user_msg=query, memory=memory)
        return str(response)

    result = asyncio.run(run_agent())
    citations = session_state["citations"]
    contexts = [c["snippet"] for c in citations if c.get("snippet")]

    print(f"Answer:\n{result}\n")
    if citations:
        print("Citations:")
        for citation in citations:
            page_text = f", page {citation['page']}" if citation.get("page") is not None else ""
            print(f"[{citation['id']}] {citation['source']}{page_text}")
        print()
    else:
        print("Citations: none\n")

    print("Evaluating...")
    faithfulness, relevancy = evaluate_with_ragas(query, result, contexts)

    print(f"{'='*60}")
    print(f"✅ Faithfulness: {faithfulness:.2f}")
    print(f"✅ Relevancy: {relevancy:.2f}")
    print(f"📊 Retrieved {len(citations)} citations")

    logger.log(
        input={"query": query, "session_id": session_id},
        output={"response": result},
        scores={"faithfulness": faithfulness, "relevancy": relevancy},
        metadata={"num_contexts": len(contexts), "num_citations": len(citations), "status": "evaluated"},
    )
    print("✅ Logged to Braintrust\n")

    if faithfulness < 0.5:
        print("⚠️  LOW FAITHFULNESS - possible hallucination")
    if relevancy < 0.5:
        print("⚠️  LOW RELEVANCY - answer may not address question")

    return result, session_id


if __name__ == "__main__":
    sid = None
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        result, sid = ask_question(query, session_id=sid)
