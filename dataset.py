# dataset.py
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

from langsmith import Client


def main():
    client = Client()

    # create a dataset in LangSmith
    dataset = client.create_dataset(
        dataset_name="Sample dataset",
        description="A sample dataset for the LangSmith hands-on tutorial",
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
