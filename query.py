import os
import uuid
import asyncio
import re
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
from outlooktool import make_email_tool, extract_citations_from_response, send_email
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

# Session store: session_id -> (memory, query_engine, agent, session_state)
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
                    "Answer questions about company policies, benefits, and procedures accurately using retrieved context provided to you. "
                    "If a question is outside the scope of HR policies, offer to email HR directly. "
                    "If the user asks to send an email to HR, you must call the send_email tool immediately using the drafted content. "
                    "Do not write out the email as a response, call the tool. "
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

    session_state = {"citations": [], "tool_events": []}

    def save_tool_event(event):
        session_state["tool_events"].append(event)

    email_tool = make_email_tool(save_tool_event=save_tool_event)
    agent = ReActAgent(
        tools=[email_tool],
        llm=llm,
        verbose=True,
    )

    sessions[session_id] = (memory, query_engine, agent, session_state)
    return memory, query_engine, agent, session_state


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


def compute_tool_metrics(tool_events: list[dict]) -> dict:
    total_calls = len(tool_events)
    email_calls = sum(1 for event in tool_events if event.get("tool") == "send_email")
    args_valid_calls = sum(1 for event in tool_events if event.get("args_valid"))
    successful_calls = sum(1 for event in tool_events if event.get("success"))
    failed_calls = total_calls - successful_calls

    arg_valid_rate = args_valid_calls / total_calls if total_calls else 1.0
    tool_success_rate = successful_calls / total_calls if total_calls else 1.0

    return {
        "tool_calls_total": total_calls,
        "email_tool_calls": email_calls,
        "tool_args_valid_calls": args_valid_calls,
        "tool_args_invalid_calls": total_calls - args_valid_calls,
        "tool_success_calls": successful_calls,
        "tool_failed_calls": failed_calls,
        "tool_args_valid_rate": arg_valid_rate,
        "tool_success_rate": tool_success_rate,
    }


def has_sufficient_context(citations: list[dict], min_score: float = 0.35) -> bool:
    if not citations:
        return False

    scored = [c.get("score") for c in citations if isinstance(c.get("score"), (int, float))]
    if not scored:
        return False
    return max(scored) >= min_score


def should_send_email(query: str) -> bool:
    text = (query or "").lower()
    has_email_word = "email" in text
    has_hr_target = "hr" in text
    has_send_intent = bool(
        re.search(r"\b(send|write|draft|submit|contact|reach out|forward)\b", text)
    )
    return has_email_word and has_hr_target and has_send_intent


def is_affirmative(text: str) -> bool:
    return (text or "").strip().lower() in {"yes", "y", "sure", "ok", "okay", "please do", "go ahead"}


def is_negative(text: str) -> bool:
    return (text or "").strip().lower() in {"no", "n", "nope", "not now", "cancel"}


def parse_name_and_topic(text: str) -> tuple[str | None, str | None]:
    value = (text or "").strip()
    name_match = re.search(r"name\s*[:\-]\s*(.+?)(?:,|;|\n|topic\s*[:\-]|$)", value, flags=re.IGNORECASE)
    topic_match = re.search(r"topic\s*[:\-]\s*(.+?)(?:,|;|\n|$)", value, flags=re.IGNORECASE)

    name = name_match.group(1).strip() if name_match else None
    topic = topic_match.group(1).strip() if topic_match else None
    return name or None, topic or None


def indicates_missing_context(answer: str) -> bool:
    text = (answer or "").lower()
    missing_markers = [
        "no information",
        "not in the provided context",
        "not mentioned in the provided context",
        "does not mention",
        "not available in the context",
        "i don't have information",
        "cannot find",
        "not found in the provided context",
    ]
    return any(marker in text for marker in missing_markers)


def ask_question(query: str, session_id: str = None):
    # Generate a session ID if one isn't provided
    if session_id is None:
        session_id = str(uuid.uuid4())
        print(f"New session started: {session_id}")

    memory, query_engine, agent, session_state = get_or_create_session(session_id)
    if "pending_email_offer" not in session_state:
        session_state["pending_email_offer"] = None
    if "pending_email_details" not in session_state:
        session_state["pending_email_details"] = None

    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Question: {query}")
    print(f"{'='*60}\n")

    session_state["tool_events"] = []

    pending_details = session_state.get("pending_email_details")
    if pending_details:
        if pending_details.get("stage") == "awaiting_details":
            name, topic = parse_name_and_topic(query)
            if not name or not topic:
                result = (
                    "Please provide both fields in this format:\n"
                    "Name: Your Name\n"
                    "Topic: Brief topic for HR"
                )
                print(f"Answer:\n{result}\n")
                return result, session_id

            pending_details["user_name"] = name
            pending_details["topic"] = topic
            pending_details["stage"] = "awaiting_confirmation"
            session_state["pending_email_details"] = pending_details
            result = (
                "Please confirm before I send this to HR:\n"
                f"Name: {name}\n"
                f"Topic: {topic}\n"
                "Reply 'yes' to send or 'no' to cancel."
            )
            print(f"Answer:\n{result}\n")
            return result, session_id

        if pending_details.get("stage") == "awaiting_confirmation":
            if is_affirmative(query):
                original_query = pending_details.get("original_query", "User requested help.")
                context_text = pending_details.get("context_text", "No additional context available.")
                user_name = pending_details.get("user_name", "Unknown")
                topic = pending_details.get("topic", "General inquiry")

                async def run_agent_email_confirmation():
                    agent_input = (
                        "The user confirmed to send an email to HR. "
                        "Call send_email now.\n\n"
                        f"Name: {user_name}\n"
                        f"Topic: {topic}\n"
                        f"Original user request:\n{original_query}\n\n"
                        f"Retrieved context:\n{context_text}\n\n"
                        "Use a concise subject and body."
                    )
                    response = await agent.run(user_msg=agent_input, memory=memory)
                    return str(response)

                result = asyncio.run(run_agent_email_confirmation())
                email_sent = any(
                    event.get("tool") == "send_email" and event.get("success")
                    for event in session_state["tool_events"]
                )
                if not email_sent:
                    fallback_subject = f"HR follow-up: {topic}"
                    fallback_body = (
                        "Hello HR,\n\n"
                        "A user requested additional assistance.\n\n"
                        f"Name: {user_name}\n"
                        f"Topic: {topic}\n"
                        f"Original request: {original_query}\n\n"
                        f"Related context:\n{context_text}\n\n"
                        "Please follow up directly.\n"
                    )
                    fallback_result = send_email(
                        subject=fallback_subject,
                        email_body=fallback_body,
                        recipient="hr@murrayosorio.com",
                    )
                    session_state["tool_events"].append(
                        {
                            "tool": "send_email",
                            "args_valid": True,
                            "success": fallback_result.lower().startswith("email sent successfully"),
                            "error": None if fallback_result.lower().startswith("email sent successfully") else fallback_result,
                            "recipient": "hr@murrayosorio.com",
                            "fallback_invoked": True,
                        }
                    )
                    result = f"{result}\n\nFallback email execution: {fallback_result}"

                session_state["pending_email_details"] = None
                tool_events = session_state["tool_events"]
                tool_metrics = compute_tool_metrics(tool_events)
                contexts = []

                print(f"Answer:\n{result}\n")
                print("Evaluating...")
                faithfulness, relevancy = evaluate_with_ragas(query, result, contexts)
                print(f"{'='*60}")
                print(f"Faithfulness: {faithfulness:.2f}")
                print(f"Relevancy: {relevancy:.2f}")
                print(f"Retrieved 0 citations")
                print(f"Tool calls: {tool_metrics['tool_calls_total']}")
                print(f"Tool success rate: {tool_metrics['tool_success_rate']:.2f}")
                print(f"Tool arg-valid rate: {tool_metrics['tool_args_valid_rate']:.2f}")
                logger.log(
                    input={"query": query, "session_id": session_id},
                    output={"response": result},
                    scores={
                        "faithfulness": faithfulness,
                        "relevancy": relevancy,
                        "tool_success_rate": tool_metrics["tool_success_rate"],
                        "tool_arg_valid_rate": tool_metrics["tool_args_valid_rate"],
                    },
                    metadata={
                        "num_contexts": 0,
                        "num_citations": 0,
                        "status": "email_sent_after_confirmation",
                        "tool_events": tool_events,
                        **tool_metrics,
                    },
                )
                print("✅ Logged to Braintrust\n")
                return result, session_id

            if is_negative(query):
                session_state["pending_email_details"] = None
                result = "Understood. I canceled the HR email request."
                print(f"Answer:\n{result}\n")
                return result, session_id

            result = "Please reply 'yes' to send the email or 'no' to cancel."
            print(f"Answer:\n{result}\n")
            return result, session_id

    pending_offer = session_state.get("pending_email_offer")
    if pending_offer:
        if is_affirmative(query):
            session_state["pending_email_details"] = {
                "stage": "awaiting_details",
                "original_query": pending_offer.get("original_query", query),
                "context_text": pending_offer.get("context_text", "No additional context available."),
            }
            session_state["pending_email_offer"] = None
            result = (
                "Please share the details before I send to HR in this format:\n"
                "Name: Your Name\n"
                "Topic: Brief topic for HR"
            )
            print(f"Answer:\n{result}\n")
            return result, session_id

        if is_negative(query):
            session_state["pending_email_offer"] = None
            result = "Understood. I won't email HR. Feel free to ask another question."
            print(f"Answer:\n{result}\n")
            return result, session_id

    retrieval_response = query_engine.query(query)
    citations = extract_citations_from_response(retrieval_response)
    session_state["citations"] = citations
    retrieved_answer = str(retrieval_response)
    contexts = [c["snippet"] for c in citations if c.get("snippet")]
    context_text = "\n".join(
        f"[{idx + 1}] {snippet}" for idx, snippet in enumerate(contexts[:5])
    )

    requires_email = should_send_email(query)
    has_context = has_sufficient_context(citations) and not indicates_missing_context(retrieved_answer)

    if not requires_email and not has_context:
        session_state["pending_email_offer"] = {
            "original_query": query,
            "context_text": context_text or "No additional context available.",
        }
        result = (
            "I couldn't find enough information in the HR documents to answer confidently. "
            "Would you like me to email HR on your behalf?"
        )
        tool_events = session_state["tool_events"]
        tool_metrics = compute_tool_metrics(tool_events)
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
        print(f"Faithfulness: {faithfulness:.2f}")
        print(f"Relevancy: {relevancy:.2f}")
        print(f"Retrieved {len(citations)} citations")
        print(f"Tool calls: {tool_metrics['tool_calls_total']}")
        print(f"Tool success rate: {tool_metrics['tool_success_rate']:.2f}")
        print(f"Tool arg-valid rate: {tool_metrics['tool_args_valid_rate']:.2f}")
        logger.log(
            input={"query": query, "session_id": session_id},
            output={"response": result},
            scores={
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "tool_success_rate": tool_metrics["tool_success_rate"],
                "tool_arg_valid_rate": tool_metrics["tool_args_valid_rate"],
            },
            metadata={
                "num_contexts": len(contexts),
                "num_citations": len(citations),
                "status": "awaiting_email_confirmation",
                "tool_events": tool_events,
                **tool_metrics,
            },
        )
        print("✅ Logged to Braintrust\n")
        return result, session_id

    if requires_email:
        session_state["pending_email_details"] = {
            "stage": "awaiting_details",
            "original_query": query,
            "context_text": context_text or "No additional context available.",
        }
        result = (
            "Before I send an email to HR, please provide:\n"
            "Name: Your Name\n"
            "Topic: Brief topic for HR"
        )
    else:
        result = retrieved_answer

    tool_events = session_state["tool_events"]
    tool_metrics = compute_tool_metrics(tool_events)

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
    print(f"Faithfulness: {faithfulness:.2f}")
    print(f"Relevancy: {relevancy:.2f}")
    print(f"Retrieved {len(citations)} citations")
    print(f"Tool calls: {tool_metrics['tool_calls_total']}")
    print(f"Tool success rate: {tool_metrics['tool_success_rate']:.2f}")
    print(f"Tool arg-valid rate: {tool_metrics['tool_args_valid_rate']:.2f}")

    logger.log(
        input={"query": query, "session_id": session_id},
        output={"response": result},
        scores={
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "tool_success_rate": tool_metrics["tool_success_rate"],
            "tool_arg_valid_rate": tool_metrics["tool_args_valid_rate"],
        },
        metadata={
            "num_contexts": len(contexts),
            "num_citations": len(citations),
            "status": "evaluated",
            "tool_events": tool_events,
            **tool_metrics,
        },
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
