import asyncio
import uuid
import os
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import Memory, StaticMemoryBlock
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.agent.workflow import ReActAgent
from outlooktool import make_email_tool, extract_citations_from_response, send_email

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
                    "If a question is outside HR policy documents, offer to email HR directly. "
                    "If the user asks to send an email to HR, call the send_email tool. "
                    "Use retrieved context for grounding."
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

    memory, query_engine, agent, session_state = get_or_create_session(session_id)
    if "pending_email_offer" not in session_state:
        session_state["pending_email_offer"] = None
    if "pending_email_details" not in session_state:
        session_state["pending_email_details"] = None

    async def run_agent():
        session_state["tool_events"] = []
        pending_details = session_state.get("pending_email_details")
        if pending_details:
            if pending_details.get("stage") == "awaiting_details":
                name, topic = parse_name_and_topic(question)
                if not name or not topic:
                    return (
                        "Please provide both fields in this format:\n"
                        "Name: Your Name\n"
                        "Topic: Brief topic for HR",
                        [],
                        session_state["tool_events"],
                        False,
                        "awaiting_email_details",
                    )

                pending_details["user_name"] = name
                pending_details["topic"] = topic
                pending_details["stage"] = "awaiting_confirmation"
                session_state["pending_email_details"] = pending_details
                return (
                    "Please confirm before I send this to HR:\n"
                    f"Name: {name}\n"
                    f"Topic: {topic}\n"
                    "Reply 'yes' to send or 'no' to cancel.",
                    [],
                    session_state["tool_events"],
                    False,
                    "awaiting_email_confirmation",
                )

            if pending_details.get("stage") == "awaiting_confirmation":
                if is_affirmative(question):
                    context_text = pending_details.get("context_text", "No additional context available.")
                    original_query = pending_details.get("original_query", "User requested help.")
                    user_name = pending_details.get("user_name", "Unknown")
                    topic = pending_details.get("topic", "General inquiry")
                    agent_input = (
                        "The user confirmed to send an email to HR. "
                        "Call send_email now.\n\n"
                        f"Name: {user_name}\n"
                        f"Topic: {topic}\n"
                        f"Original user request:\n{original_query}\n\n"
                        f"Retrieved context:\n{context_text}\n\n"
                        "Use a concise subject and body."
                    )
                    agent_response = await agent.run(user_msg=agent_input, memory=memory)
                    final_answer = str(agent_response)
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
                        final_answer = f"{final_answer}\n\nFallback email execution: {fallback_result}"
                    session_state["pending_email_details"] = None
                    return final_answer, [], session_state["tool_events"], False, None

                if is_negative(question):
                    session_state["pending_email_details"] = None
                    return "Understood. I canceled the HR email request.", [], session_state["tool_events"], False, None

                return "Please reply 'yes' to send the email or 'no' to cancel.", [], session_state["tool_events"], False, "awaiting_email_confirmation"

        pending_offer = session_state.get("pending_email_offer")
        if pending_offer:
            if is_affirmative(question):
                session_state["pending_email_details"] = {
                    "stage": "awaiting_details",
                    "original_query": pending_offer.get("original_query", question),
                    "context_text": pending_offer.get("context_text", "No additional context available."),
                }
                session_state["pending_email_offer"] = None
                return (
                    "Please share the details before I send to HR in this format:\n"
                    "Name: Your Name\n"
                    "Topic: Brief topic for HR",
                    [],
                    session_state["tool_events"],
                    False,
                    "awaiting_email_details",
                )

            if is_negative(question):
                session_state["pending_email_offer"] = None
                return "Understood. I won't email HR. Feel free to ask another question.", [], session_state["tool_events"], False, None

        retrieval_response = query_engine.query(question)
        citations = extract_citations_from_response(retrieval_response)
        retrieved_answer = str(retrieval_response)
        context_snippets = [c["snippet"] for c in citations if c.get("snippet")]
        context_text = "\n".join(
            f"[{idx + 1}] {snippet}" for idx, snippet in enumerate(context_snippets[:5])
        )

        requires_email = should_send_email(question)
        has_context = has_sufficient_context(citations) and not indicates_missing_context(retrieved_answer)

        if not requires_email and not has_context:
            session_state["pending_email_offer"] = {
                "original_query": question,
                "context_text": context_text or "No additional context available.",
            }
            return (
                "I couldn't find enough information in the HR documents to answer confidently. "
                "Would you like me to email HR on your behalf?",
                citations,
                session_state["tool_events"],
                True,
                "awaiting_email_opt_in",
            )

        if requires_email:
            session_state["pending_email_details"] = {
                "stage": "awaiting_details",
                "original_query": question,
                "context_text": context_text or "No additional context available.",
            }
            final_answer = (
                "Before I send an email to HR, please provide:\n"
                "Name: Your Name\n"
                "Topic: Brief topic for HR"
            )
            return final_answer, citations, session_state["tool_events"], False, "awaiting_email_details"
        else:
            final_answer = retrieved_answer

        return final_answer, citations, session_state["tool_events"], False, None

    try:
        answer, citations, tool_events, needs_email_confirmation, email_workflow_stage = asyncio.run(run_agent())
        tool_metrics = compute_tool_metrics(tool_events)
        return jsonify(
            {
                'answer': answer,
                'session_id': session_id,
                'citations': citations,
                'tool_metrics': tool_metrics,
                'tool_events': tool_events,
                'needs_email_confirmation': needs_email_confirmation,
                'email_workflow_stage': email_workflow_stage,
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
