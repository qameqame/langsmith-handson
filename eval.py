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
