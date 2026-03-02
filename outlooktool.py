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


def make_search_tool(query_engine):
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
        if hasattr(query_engine, "query"):
            response = query_engine.query(question)
        elif hasattr(query_engine, "chat"):
            response = query_engine.chat(question)
        else:
            raise ValueError("Search tool expected an engine with .query() or .chat()")
        return str(response)

    return FunctionTool.from_defaults(fn=search_hr_docs)


email_tool = FunctionTool.from_defaults(fn=send_email)
