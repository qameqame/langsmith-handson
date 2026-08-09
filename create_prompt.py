# create_prompt.py
from dotenv import load_dotenv

load_dotenv()  # .env から環境変数を読み込む

from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

client = Client()

prompt = ChatPromptTemplate(
    [
        ("system", "あなたは親切なチャットボットです。"),
        ("user", "{question}"),
    ]
)

client.push_prompt("prompt-quickstart", object=prompt)
