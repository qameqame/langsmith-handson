English | [日本語](README.ja.md)

# LangSmith Hands-On Tutorial (Python)

This repository is a hands-on project for learning LangSmith by doing. You'll walk through three parts in order:

1. **Tracing** — visualize what happens inside your LLM app at runtime
2. **Evaluation** — quantitatively measure the quality of your LLM app's output
3. **Prompt management (Prompt Hub)** — version and share prompts

Expect to spend 45-60 minutes. Basic familiarity with Python is assumed.

## Repository layout

```
langsmith-handson/
├── .env.example       # environment variable template (safe to commit)
├── .gitignore         # excludes .env and friends from Git
├── requirements.txt   # dependencies
├── app.py             # Part 1: tracing sample app
├── dataset.py          # Part 2: create the evaluation dataset
├── eval.py             # Part 2: target function, evaluator, and run
├── create_prompt.py    # Part 3: create a prompt and push it to Prompt Hub
├── test_prompt.py      # Part 3: pull and run a prompt
└── iterate_prompt.py   # Part 3: update a prompt (new commit)
```

`.env` itself is not included in this repository (excluded via `.gitignore`). Create your own and set your API keys.

---

## 0. Prerequisites

Before you start, make sure you have:

- **A LangSmith account**: sign up at [smith.langchain.com](https://smith.langchain.com) (a free plan is available)
- **A LangSmith API key**: issued from LangSmith's `Settings` → `API Keys`
- **An Anthropic API key**: issued from [console.anthropic.com](https://console.anthropic.com/settings/keys) (this tutorial uses Anthropic (Claude) as the LLM provider)
- **Python 3.9 or later**

### Setup

Instead of hardcoding API keys with shell `export`, this project writes them to a `.env` file and loads them with `python-dotenv`. `.env` is excluded via `.gitignore`, so there's no risk of accidentally committing your keys to the repository.

```bash
cd langsmith-handson
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure environment variables (.env file)

Copy `.env.example` to `.env` and replace the values with your own API keys.

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

Each Python script starts by calling `load_dotenv()`, which automatically loads the contents of `.env` into environment variables.

```python
from dotenv import load_dotenv
load_dotenv()
```

You're all set. Let's walk through the three parts.

---

## Part 1. Tracing

The most fundamental feature of LangSmith is the "trace." It records the full sequence of an LLM call, tool execution, and everything in between, so you can visualize it in the LangSmith UI.

### 1-1. The sample app (`app.py`)

This app is a simple assistant: it retrieves context for a question, then has the LLM answer using that context.

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

There are two wrappers to note:

- `wrap_anthropic(anthropic.Anthropic())`: wraps the Anthropic client so every LLM call is automatically logged as a nested span.
- `@traceable`: wraps a function so its inputs, outputs, and any nested calls are captured as a single trace. Passing `run_type="tool"` records it as a tool call.

> **Note**: Anthropic's Messages API differs from OpenAI's — the `system` prompt is passed as a dedicated `system` parameter rather than inside the `messages` array, and the response text is read from `message.content[0].text`.

### 1-2. Run it

```bash
python3 app.py
```

### 1-3. View the trace in the LangSmith UI

1. Open the [LangSmith UI](https://smith.langchain.com) and go to **Tracing** in the left menu.
2. Select the **default** project (traces go here automatically if you haven't set `LANGSMITH_PROJECT`).
3. Click the `assistant` row to open the trace.
4. On the **Details** tab, confirm that `get_context` (the tool call) and the Anthropic call appear as a nested tree inside the `assistant` function.

Now you can see exactly what ran, in what order, and with what inputs and outputs, inside your LLM app.

> **Tip**: To send traces to a specific project, set `LANGSMITH_PROJECT="my-project-name"` in `.env`.

---

## Part 2. Evaluation

Tracing shows you what happened. Next, it's time to measure output quality quantitatively. LangSmith's evaluation feature has three components:

- **Dataset**: a set of test inputs (and, if available, expected outputs)
- **Target function**: the application logic you want to evaluate
- **Evaluators**: functions that score the output

### 2-1. Create a dataset (`dataset.py`)

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

Run it.

```bash
python3 dataset.py
```

### 2-2. Define the target function (`eval.py`)

Define the logic you want to evaluate — here, a simple LLM call that answers a question.

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

### 2-3. Define an evaluator

We use `CORRECTNESS_PROMPT` from the `openevals` library — an LLM-as-judge, meaning the LLM itself scores the output.

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

> Passing a `provider:model` string like `model="anthropic:claude-sonnet-4-6"` tells `openevals` to initialize a LangChain chat model internally. This requires the `langchain-anthropic` package (already included in `requirements.txt`).

The evaluator compares three things:

- `inputs`: what was passed into the target function (the question)
- `outputs`: what the target function returned (the model's answer)
- `reference_outputs`: the expected answer registered in the dataset (see 2-1)

### 2-4. Run the evaluation

`Client.evaluate(...)` runs the experiment.

```python
def main():
    client = Client()
    experiment_results = client.evaluate(
        target,
        data="Sample dataset",
        evaluators=[
            correctness_evaluator,
            # you can add multiple evaluators here
        ],
        experiment_prefix="first-eval-in-langsmith",
        max_concurrency=2,
    )
    print(experiment_results)

if __name__ == "__main__":
    main()
```

Run it.

```bash
python3 eval.py
```

After running, a link like this is printed to the terminal:

```
View the evaluation results for experiment: 'first-eval-in-langsmith-xxxxxxxx' at:
https://smith.langchain.com/.../compare?selectedSessions=...
```

Open the link to see a table on the **Datasets & Experiments** page showing each test case's `Inputs`, `Reference Output`, `Outputs`, and correctness score. This lets you quantitatively compare whether quality improved or regressed each time you change a prompt or model.

---

## Part 3. Prompt management (Prompt Hub)

Finally, let's use Prompt Hub — a mechanism for versioning and sharing prompts independently of your code.

### 3-1. Create and push a prompt (`create_prompt.py`)

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

Run it.

```bash
python3 create_prompt.py
```

Follow the printed link to confirm that a new prompt (`prompt-quickstart`) now appears in the **Prompts** section of the LangSmith UI.

### 3-2. Pull and use a prompt (`test_prompt.py`)

Pull the prompt you just pushed and use it in an actual LLM call.

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

> For Anthropic, we use `ChatAnthropic` from `langchain-anthropic` and connect the prompt and model directly with LangChain's `prompt | model` runnable syntax. Unlike using the raw OpenAI SDK, there's no need to write your own message-format conversion (the equivalent of `convert_to_openai_messages`).

```bash
python3 test_prompt.py
```

> To pin a specific version, pass a commit hash like `client.pull_prompt("prompt-quickstart:<commit-hash>")`. This is the recommended way to ensure reproducibility in production.

### 3-3. Update a prompt (commit a new version) (`iterate_prompt.py`)

Calling `push_prompt` again with the same prompt name adds a new commit to its history (previous versions are preserved).

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

Open `Prompts` → `prompt-quickstart` in the LangSmith UI to see two commits in the history. Workspace members can also edit prompts directly in the Playground and save changes as new commits.

---

## Summary: how the three features connect

| Feature | Role | Files |
| --- | --- | --- |
| Tracing | Visualize and debug what's happening inside a run | `app.py` |
| Evaluation | Quantitatively measure and compare output quality | `dataset.py`, `eval.py` |
| Prompt management | Version and share prompts | `create_prompt.py`, `test_prompt.py`, `iterate_prompt.py` |

A typical real-world workflow: manage prompts in Prompt Hub, validate quality with the Playground and evaluations, and continuously monitor actual behavior in production with tracing.

## Next steps

- **LangChain/LangGraph integration**: if you're using LangChain or LangGraph, tracing can be enabled with just an environment variable.
- **Online evaluation**: run LLM-as-judge automatically against production traffic to continuously monitor quality.
- **Custom evaluators**: beyond the built-in evaluators in `openevals`, you can define evaluation logic in arbitrary Python code.
- **LangSmith CLI**: inspect traces directly from the terminal.

## References

- [Tracing quickstart](https://docs.langchain.com/langsmith/observability-quickstart)
- [Evaluation quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart)
- [Prompt engineering quickstart](https://docs.langchain.com/langsmith/prompt-engineering-quickstart)
- [LangSmith official site](https://smith.langchain.com)
