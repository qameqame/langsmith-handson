# test_prompt.py
from dotenv import load_dotenv

load_dotenv()  # .env から環境変数を読み込む

from langsmith import Client
from langchain_anthropic import ChatAnthropic

client = Client()
model = ChatAnthropic(model="claude-sonnet-4-6")

# 最新バージョンのプロンプトを取得
prompt = client.pull_prompt("prompt-quickstart")

# プロンプトとモデルをつないでそのまま呼び出す
chain = prompt | model
response = chain.invoke({"question": "空はなぜ青いのですか？"})
print(response.content)
