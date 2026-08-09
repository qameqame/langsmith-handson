[English](README.md) | 日本語

# LangSmith ハンズオンチュートリアル（Python）

このリポジトリは、LangSmithを実際に手を動かしながら学ぶためのハンズオン用プロジェクトです。以下の3つを順番に体験します。

1. トレーシング（Tracing）— LLMアプリの実行内容を可視化する
2. 評価（Evaluation）— LLMアプリの出力品質を定量的に測定する
3. プロンプト管理（Prompt Hub）— プロンプトをバージョン管理・共有する

所要時間の目安は45〜60分です。Pythonの基本文法がわかることを前提としています。

## リポジトリ構成

```
langsmith-handson/
├── .env.example       # 環境変数のテンプレート（コミットする）
├── .gitignore         # .env などをGit管理から除外
├── requirements.txt   # 依存パッケージ
├── app.py             # Part 1: トレーシングのサンプルアプリ
├── dataset.py          # Part 2: 評価用データセットの作成
├── eval.py             # Part 2: 評価対象関数・評価者・評価の実行
├── create_prompt.py    # Part 3: プロンプトの作成とPrompt Hubへのプッシュ
├── test_prompt.py      # Part 3: プロンプトの取得と実行
└── iterate_prompt.py   # Part 3: プロンプトの更新（新しいコミット）
```

`.env` 自体はこのリポジトリに含まれていません（`.gitignore` で除外）。各自で作成してAPIキーを設定してください。

---

## 0. 事前準備

始める前に、以下を用意してください。

- **LangSmithアカウント**：[smith.langchain.com](https://smith.langchain.com) でサインアップ（無料プランあり）
- **LangSmith APIキー**：LangSmithの `Settings` → `API Keys` から発行
- **Anthropic APIキー**：[console.anthropic.com](https://console.anthropic.com/settings/keys) から発行（本チュートリアルはAnthropic（Claude）をLLMプロバイダーとして使用します）
- **Python 3.9以上**

### セットアップ

APIキーはシェルの `export` に直書きせず、`.env` ファイルに書いて `python-dotenv` で読み込みます。`.env` は `.gitignore` で除外されるため、キーが誤ってリポジトリにコミットされる心配がありません。

```bash
cd langsmith-handson
python3 -m venv .venv
source .venv/bin/activate  # Windowsの場合は .venv\Scripts\activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 環境変数の設定（.envファイル）

`.env.example` をコピーして `.env` を作成し、値を自分のAPIキーに置き換えます。

```bash
cp .env.example .env
```

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=langsmith-handson

# Only set this if you're on a non-US region (EU/APAC, etc.)
# LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com

ANTHROPIC_API_KEY=your-anthropic-api-key
```

各Pythonスクリプトの冒頭では以下のように `load_dotenv()` を呼んでおり、`.env` の内容が自動的に環境変数として読み込まれます。

```python
from dotenv import load_dotenv
load_dotenv()
```

これで準備は完了です。ここから3つのパートを順に進めます。

---

## Part 1. トレーシング（Tracing）

LangSmithの最も基本的な機能が「トレース」です。LLM呼び出しやツール実行など、リクエストの一連の流れを記録し、LangSmithのUI上で可視化できます。

### 1-1. サンプルアプリ（`app.py`）

このアプリは「質問に対してコンテキストを取得し、それを踏まえてLLMが回答する」という簡単なアシスタントです。

```python
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
```

ポイントは2つのラッパーです。

- `wrap_anthropic(anthropic.Anthropic())`：Anthropicクライアントをラップし、すべてのLLM呼び出しを自動でネストされたスパンとしてログに記録します。
- `@traceable`：関数をラップし、その入出力とネストされた処理を1つのトレースとしてまとめます。`run_type="tool"` を指定すると、ツール呼び出しとして種別が記録されます。

> **Note**：Anthropicの Messages API はOpenAIと異なり、`system` プロンプトを `messages` 配列ではなく専用の `system` パラメータとして渡します。応答本文も `message.content[0].text` から取得します。

### 1-2. 実行する

```bash
python3 app.py
```

### 1-3. LangSmith UIでトレースを確認する

1. [LangSmith UI](https://smith.langchain.com) を開き、左メニューの **Tracing** に移動します。
2. **default** プロジェクト（`LANGSMITH_PROJECT` を設定していない場合は自動でここに送られます）を選択します。
3. `assistant` という行をクリックしてトレースを開きます。
4. **Details** タブで、`assistant` 関数の中に `get_context`（ツール呼び出し）と Anthropic呼び出しがネストされたツリー構造として表示されることを確認します。

これでLLMアプリの内部で「何が」「どの順番で」「どんな入出力で」実行されたかが一目でわかるようになりました。

> **Tips**：特定のプロジェクトにトレースを送りたい場合は、`.env` に `LANGSMITH_PROJECT="my-project-name"` を設定してください。

---

## Part 2. 評価（Evaluation）

トレーシングで「何が起きたか」は見えるようになりましたが、次は「出力の品質」を定量的に測る番です。LangSmithの評価機能は次の3要素で構成されます。

- **Dataset（データセット）**：テスト用の入力（と、あれば正解の出力）の集合
- **Target function（対象関数）**：評価したいアプリのロジック本体
- **Evaluators（評価者）**：出力を採点する関数

### 2-1. データセットを作成する（`dataset.py`）

```python
# dataset.py
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

from langsmith import Client

def main():
    client = Client()

    # create a dataset in LangSmith
    dataset = client.create_dataset(
        dataset_name="Sample dataset",
        description="A sample dataset for the LangSmith hands-on tutorial"
    )

    # define test cases (input / expected output pairs)
    examples = [
        {
            "inputs": {"question": "Which country is Mount Kilimanjaro located in?"},
            "outputs": {"answer": "Mount Kilimanjaro is located in Tanzania."},
        },
        {
            "inputs": {"question": "What is Earth's lowest point?"},
            "outputs": {"answer": "Earth's lowest point is the Dead Sea."},
        },
    ]

    client.create_examples(dataset_id=dataset.id, examples=examples)
    print("Created dataset:", dataset.name)

if __name__ == "__main__":
    main()
```

実行します。

```bash
python3 dataset.py
```

### 2-2. 評価対象の関数（Target function）（`eval.py`）

評価したいロジック（ここでは質問に答える単純なLLM呼び出し）を定義します。

```python
# eval.py
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

import anthropic
from langsmith import Client
from langsmith.wrappers import wrap_anthropic
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

# wrap the Anthropic client to enable tracing
anthropic_client = wrap_anthropic(anthropic.Anthropic())

# the application logic you want to evaluate; the dataset's inputs are passed in automatically
def target(inputs: dict) -> dict:
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="Answer the following question accurately",
        messages=[
            {"role": "user", "content": inputs["question"]},
        ],
    )
    return {"answer": message.content[0].text.strip()}
```

### 2-3. 評価者（Evaluator）

`openevals` ライブラリの `CORRECTNESS_PROMPT`（LLM-as-judge、つまりLLM自身に採点させる仕組み）を使います。

```python
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="anthropic:claude-sonnet-4-6",
        feedback_key="correctness",
    )
    return evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs,
    )
```

> `model="anthropic:claude-sonnet-4-6"` のように `provider:model` 形式の文字列を渡すと、`openevals` が内部でLangChainのチャットモデルを初期化します。これには `langchain-anthropic` パッケージが必要です（`requirements.txt` に含まれています）。

評価者は次の3つを比較して採点します。

- `inputs`：対象関数に渡された入力（質問文）
- `outputs`：対象関数が返した出力（モデルの回答）
- `reference_outputs`：データセットに登録した正解（2-1参照）

### 2-4. 評価を実行する

`Client.evaluate(...)` で実験を実行します。

```python
def main():
    client = Client()
    experiment_results = client.evaluate(
        target,
        data="Sample dataset",
        evaluators=[
            correctness_evaluator,
            # 複数の評価者を並べて追加することもできる
        ],
        experiment_prefix="first-eval-in-langsmith",
        max_concurrency=2,
    )
    print(experiment_results)

if __name__ == "__main__":
    main()
```

実行します。

```bash
python3 eval.py
```

実行後、以下のようなリンクがターミナルに出力されます。

```
View the evaluation results for experiment: 'first-eval-in-langsmith-xxxxxxxx' at:
https://smith.langchain.com/.../compare?selectedSessions=...
```

このリンクを開くと、**Datasets & Experiments** ページで各テストケースの `Inputs`・`Reference Output`・`Outputs`・スコア（correctness）が表形式で確認できます。これにより、プロンプトやモデルを変更するたびに「品質が上がったか下がったか」を定量的に比較できるようになります。

---

## Part 3. プロンプト管理（Prompt Hub）

最後に、プロンプトをコードから切り離してバージョン管理・共有する仕組みであるPrompt Hubを使います。

### 3-1. プロンプトを作成してプッシュする（`create_prompt.py`）

```python
# create_prompt.py
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

client = Client()

prompt = ChatPromptTemplate([
    ("system", "You are a helpful chatbot."),
    ("user", "{question}"),
])

client.push_prompt("prompt-quickstart", object=prompt)
```

実行します。

```bash
python3 create_prompt.py
```

出力されたリンクを開くと、LangSmith UIの **Prompts** セクションに新しいプロンプト（`prompt-quickstart`）が作成されているのが確認できます。

### 3-2. プロンプトを取得して使う（`test_prompt.py`）

先ほどプッシュしたプロンプトを取得（pull）して実際にLLM呼び出しに使います。

```python
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
```

> Anthropicでは `langchain-anthropic` の `ChatAnthropic` を使い、`prompt | model` というLangChainのRunnable構文でプロンプトとモデルを直接つないでいます。OpenAIの生SDKを使う場合と違い、メッセージ形式の変換（`convert_to_openai_messages` 相当の処理）を自分で書く必要がありません。

```bash
python3 test_prompt.py
```

> 特定バージョンを固定したい場合は `client.pull_prompt("prompt-quickstart:<commit-hash>")` のようにコミットハッシュを指定します。本番環境ではこの方法で再現性を確保するのが推奨されます。

### 3-3. プロンプトを更新（新しいバージョンをコミット）する（`iterate_prompt.py`）

同じプロンプト名で再度 `push_prompt` を呼ぶと、新しいコミットとして履歴に追加されます（過去のバージョンは失われません）。

```python
# iterate_prompt.py
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

client = Client()

new_prompt = ChatPromptTemplate([
    ("system", "You are a helpful chatbot. Always respond in a formal, professional tone."),
    ("user", "{question}"),
])

client.push_prompt("prompt-quickstart", object=new_prompt)
```

```bash
python3 iterate_prompt.py
```

LangSmith UIの `Prompts` → `prompt-quickstart` を開くと、2つのコミット履歴が確認できます。チームメンバーはPlayground上でも直接プロンプトを編集し、新しいコミットとして保存できます。

---

## まとめ：3つの機能のつながり

| 機能 | 役割 | 対応ファイル |
| --- | --- | --- |
| トレーシング | 実行の中身を可視化しデバッグする | `app.py` |
| 評価 | 出力品質を定量的に測定・比較する | `dataset.py`, `eval.py` |
| プロンプト管理 | プロンプトをバージョン管理・共有する | `create_prompt.py`, `test_prompt.py`, `iterate_prompt.py` |

実務での典型的なワークフローは、Prompt Hubでプロンプトを管理しながらPlaygroundや評価で品質を検証し、本番環境ではトレーシングで実際の挙動を継続的に監視する、というサイクルになります。

## 次のステップ

- **LangChain/LangGraphとの統合**：LangChainやLangGraphを使っている場合、環境変数を設定するだけでトレーシングが有効になります。
- **オンライン評価**：本番トラフィックに対してLLM-as-judgeを自動実行し、継続的に品質を監視できます。
- **カスタム評価者**：`openevals` の組み込み評価者だけでなく、任意のPythonコードで評価ロジックを定義することも可能です。
- **LangSmith CLI**：ターミナルからトレースを検査することもできます。

## 参考リンク

- [Tracing quickstart](https://docs.langchain.com/langsmith/observability-quickstart)
- [Evaluation quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart)
- [Prompt engineering quickstart](https://docs.langchain.com/langsmith/prompt-engineering-quickstart)
- [LangSmith 公式サイト](https://smith.langchain.com)
