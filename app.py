# app.py
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

import anthropic
from langsmith.wrappers import wrap_anthropic
from langsmith import traceable

# wrapping the Anthropic client automatically traces every LLM call
client = wrap_anthropic(anthropic.Anthropic())


@traceable(run_type="tool")  # trace this as a tool call
def get_context(question: str) -> str:
    # in a real app this would query a vector DB or knowledge base
    return "LangSmith traces are stored for 14 days on the Developer plan."


@traceable  # record this entire function as a single trace
def assistant(question: str) -> str:
    context = get_context(question)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=f"Answer using the context below.\n\nContext: {context}",
        messages=[
            {"role": "user", "content": question},
        ],
    )
    return message.content[0].text


if __name__ == "__main__":
    print(assistant("How long are LangSmith traces stored?"))
