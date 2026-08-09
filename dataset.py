# dataset.py
from dotenv import load_dotenv

load_dotenv()  # .env から環境変数を読み込む

from langsmith import Client


def main():
    client = Client()

    # LangSmith上にデータセットを作成
    dataset = client.create_dataset(
        dataset_name="Sample dataset",
        description="LangSmithハンズオン用のサンプルデータセット",
    )

    # テストケース（入力と正解出力のペア）を定義
    examples = [
        {
            "inputs": {"question": "キリマンジャロ山はどの国にありますか？"},
            "outputs": {"answer": "キリマンジャロ山はタンザニアにあります。"},
        },
        {
            "inputs": {"question": "地球上で最も低い場所はどこですか？"},
            "outputs": {"answer": "地球上で最も低い場所は死海です。"},
        },
    ]

    client.create_examples(dataset_id=dataset.id, examples=examples)
    print("データセットを作成しました:", dataset.name)


if __name__ == "__main__":
    main()
