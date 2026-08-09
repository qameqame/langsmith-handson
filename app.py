# app.py
from dotenv import load_dotenv

load_dotenv()  # .env から環境変数を読み込む

import anthropic
from langsmith.wrappers import wrap_anthropic
from langsmith import traceable

# Anthropicクライアントをラップすると、すべてのLLM呼び出しが自動的にトレースされる
client = wrap_anthropic(anthropic.Anthropic())


@traceable(run_type="tool")  # ツール呼び出しとしてトレースする
def get_context(question: str) -> str:
    # 実際のアプリではベクトルDBやナレッジベースを検索する処理になる
    return "LangSmithのトレースはDeveloperプランで14日間保存されます。"


@traceable  # この関数全体を1つのトレースとして記録する
def assistant(question: str) -> str:
    context = get_context(question)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=f"以下のコンテキストを踏まえて回答してください。\n\nコンテキスト: {context}",
        messages=[
            {"role": "user", "content": question},
        ],
    )
    return message.content[0].text


if __name__ == "__main__":
    print(assistant("LangSmithのトレースはどのくらい保存されますか？"))
