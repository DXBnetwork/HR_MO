import os
import requests
from llama_index.core.tools import FunctionTool


def send_email(subject: str, email_body: str, recipient: str = "hr@murrayosorio.com") -> str:
    """
    Send an email via Outlook using Microsoft Graph API.
    Args:
        subject: The email subject line
        email_body: The content of the email
        recipient: The email address to send to
    """
    graph_url = "https://graph.microsoft.com/v1.0/me/sendMail"
    access_token = os.environ.get("MICROSOFT_ACCESS_TOKEN")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    email_data = {
        "message": {
            "subject": subject,
            "body": {
                "contentType": "Text",
                "content": email_body
            },
            "toRecipients": [
                {
                    "emailAddress": {
                        "address": recipient
                    }
                }
            ]
        }
    }

    response = requests.post(graph_url, headers=headers, json=email_data)
    if response.status_code == 202:
        return f"Email sent successfully to {recipient}"
    else:
        return f"Failed to send email: {response.text}"


def _extract_citations(response) -> list[dict]:
    citations = []
    source_nodes = getattr(response, "source_nodes", []) or []

    for idx, source_node in enumerate(source_nodes, start=1):
        node = getattr(source_node, "node", None)
        metadata = getattr(node, "metadata", {}) or {}
        text = getattr(source_node, "text", "") or ""

        citation = {
            "id": idx,
            "source": metadata.get("file_name")
            or metadata.get("filename")
            or metadata.get("document_id")
            or "unknown",
            "page": metadata.get("page_label")
            or metadata.get("page_number")
            or metadata.get("page"),
            "score": getattr(source_node, "score", None),
            "snippet": " ".join(text.split())[:240],
        }
        citations.append(citation)

    return citations


def extract_citations_from_response(response) -> list[dict]:
    return _extract_citations(response)


def make_search_tool(query_engine, save_citations=None, save_tool_event=None):
    """
    Wraps the query engine as a search tool.
    Calls Pinecone directly, memory lives at the agent level.
    """
    def search_hr_docs(question: str) -> str:
        """
        Search HR policy documents to answer questions about company policies,
        benefits, procedures, and workplace guidelines.
        Args:
            question: The HR-related question to search for
        """
        event = {
            "tool": "search_hr_docs",
            "args_valid": bool(question and question.strip()),
            "success": False,
            "error": None,
        }
        if not event["args_valid"]:
            event["error"] = "Missing required argument: question"
            if save_tool_event is not None:
                save_tool_event(event)
            return "Search tool failed: question is required."

        try:
            if hasattr(query_engine, "query"):
                response = query_engine.query(question)
            elif hasattr(query_engine, "chat"):
                response = query_engine.chat(question)
            else:
                raise ValueError("Search tool expected an engine with .query() or .chat()")
        except Exception as exc:
            event["error"] = str(exc)
            if save_tool_event is not None:
                save_tool_event(event)
            return f"Search tool failed: {exc}"

        citations = _extract_citations(response)
        event["success"] = True
        event["num_citations"] = len(citations)

        if save_tool_event is not None:
            save_tool_event(event)
        if save_citations is not None:
            save_citations(citations)

        if not citations:
            return f"{str(response)}\n\nCitations: none"

        citation_lines = []
        for citation in citations:
            page_text = f", page {citation['page']}" if citation["page"] is not None else ""
            citation_lines.append(f"[{citation['id']}] {citation['source']}{page_text}")

        return f"{str(response)}\n\nCitations:\n" + "\n".join(citation_lines)

    return FunctionTool.from_defaults(fn=search_hr_docs)


def make_email_tool(save_tool_event=None):
    def send_email_with_metrics(
        subject: str,
        email_body: str,
        recipient: str = "hr@murrayosorio.com",
    ) -> str:
        event = {
            "tool": "send_email",
            "args_valid": bool(subject and subject.strip() and email_body and email_body.strip()),
            "success": False,
            "error": None,
            "recipient": recipient,
        }

        if not event["args_valid"]:
            event["error"] = "Missing required arguments: subject and email_body"
            if save_tool_event is not None:
                save_tool_event(event)
            return "Email tool failed: subject and email_body are required."

        result = send_email(subject=subject, email_body=email_body, recipient=recipient)
        event["success"] = result.lower().startswith("email sent successfully")
        if not event["success"]:
            event["error"] = result

        if save_tool_event is not None:
            save_tool_event(event)
        return result

    return FunctionTool.from_defaults(fn=send_email_with_metrics)


email_tool = make_email_tool()
