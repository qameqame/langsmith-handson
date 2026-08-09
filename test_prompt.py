# test_prompt.py
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

from langsmith import Client
from langchain_anthropic import ChatAnthropic

client = Client()
model = ChatAnthropic(model="claude-sonnet-4-6")

# pull the latest version of the prompt
prompt = client.pull_prompt("prompt-quickstart")

# chain the prompt and model together and invoke it directly
chain = prompt | model
response = chain.invoke({"question": "Why is the sky blue?"})
print(response.content)
