import json
from langsmith import Client
import os
import json
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "RAG Retrieval Evaluation"




def main():

    # Load your existing evaluation questions
    with open("evaluation/evaluation_dataset.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    client = Client()

    # Create dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Evaluation dataset for hybrid RAG retrieval"
    )

    examples = []

    for q in questions:

        # Skip questions without an expected source
        if q.get("expected_source") is None:
            continue

        examples.append(
            {
                "inputs": {
                    "question": q["question"]
                },
                "outputs": {
                    "expected_source": q["expected_source"]
                }
            }
        )

    client.create_examples(
        dataset_id=dataset.id,
        examples=examples
    )

    print("=" * 60)
    print("LangSmith dataset created")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Examples: {len(examples)}")
    print("=" * 60)


if __name__ == "__main__":
    main()