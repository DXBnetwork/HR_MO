import os
import uuid
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
evaluator_llm = ChatGroq(
    model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"]
)

# Session store: session_id -> (memory, chat_engine)
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
                    "The HR email address is hr@murrayosorio.com."
                ),
                priority=0,
            )
        ],
    )

    chat_engine = index.as_chat_engine(
        llm=llm,
        memory=memory,
        chat_mode="context",
    )

    sessions[session_id] = (memory, chat_engine)
    return memory, chat_engine


def evaluate_faithfulness(answer: str, contexts: list) -> float:
    context_text = "\n\n".join(contexts[:3])
    prompt = f"""You are evaluating if an answer is grounded in the provided context.
Context:
{context_text}
Answer:
{answer}
Is the answer faithful to the context? Score from 0.0 to 1.0 where:
- 0.0 = completely made up, not in context at all
- 0.5 = partially grounded, some claims not in context
- 1.0 = fully grounded, all claims supported by context
Respond with ONLY a number between 0.0 and 1.0, nothing else."""
    try:
        response = evaluator_llm.invoke(prompt)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def evaluate_relevancy(question: str, answer: str, contexts: list) -> float:
    context_text = "\n\n".join(contexts[:3])
    prompt = f"""You are evaluating if an answer properly addresses the question using the provided context.
Context:
{context_text}
Question:
{question}
Answer:
{answer}
Evaluate two things:
1. Does the answer directly address what the question is asking?
2. Is the answer using information from the context (not making things up)?
Score from 0.0 to 1.0 where:
- 0.0 = irrelevant OR not grounded in context
- 0.5 = somewhat addresses question but incomplete or partially made up
- 1.0 = directly answers question using only context
Respond with ONLY a number between 0.0 and 1.0, nothing else."""
    try:
        response = evaluator_llm.invoke(prompt)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def ask_question(query: str, session_id: str = None):
    # Generate a session ID if one isn't provided
    if session_id is None:
        session_id = str(uuid.uuid4())
        print(f"New session started: {session_id}")

    memory, chat_engine = get_or_create_session(session_id)

    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Question: {query}")
    print(f"{'='*60}\n")

    response = chat_engine.chat(query)
    result = str(response)
    contexts = [node.text for node in response.source_nodes]

    print(f"Answer:\n{result}\n")

    print("Evaluating...")
    faithfulness = evaluate_faithfulness(result, contexts)
    relevancy = evaluate_relevancy(query, result, contexts)  # fixed: pass contexts

    print(f"{'='*60}")
    print(f"✅ Faithfulness: {faithfulness:.2f}")
    print(f"✅ Relevancy: {relevancy:.2f}")
    print(f"📊 Retrieved {len(contexts)} context chunks")

    logger.log(
        input={"query": query, "session_id": session_id},
        output={"response": result},
        scores={"faithfulness": faithfulness, "relevancy": relevancy},
        metadata={"num_contexts": len(contexts), "status": "evaluated"},
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