# iterate_prompt.py
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

client = Client()

new_prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful chatbot. Always respond in a formal, professional tone."),
        ("user", "{question}"),
    ]
)

client.push_prompt("prompt-quickstart", object=new_prompt)
