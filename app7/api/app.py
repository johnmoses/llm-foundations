from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import argparse

from rag import MilvusRAG, rag_generate
from data import (
    generate_finance_datasets,
    BASIC_FINETUNE_PATH,
    PEFT_FINETUNE_PATH,
    RLHF_PREFERENCES_PATH,
)
# from fine_tune import (
#     basic_fine_tuning,
#     peft_fine_tuning,
#     rlhf_reward_model_training,
# )
# from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


rag_client = None
model = None
tokenizer = None


@app.on_event("startup")
def startup_event():
    global rag_client, model, tokenizer
    rag_client = MilvusRAG(db_path="milvus_rag_db.db")
    # model_path = "./basic_finetune_results"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    # model.eval()


@app.post("/generate")
def generate_answer(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    answer = rag_generate(request.query, rag_client, model, tokenizer)
    return {"answer": answer}


@app.post("/generate_datasets")
def api_generate_datasets():
    generate_finance_datasets()
    return {"status": "Datasets generated successfully."}


@app.post("/basic_finetune")
def api_basic_finetune():
    # basic_fine_tuning("meta-llama/Llama-3-8b", BASIC_FINETUNE_PATH)
    return {"status": "Basic fine-tuning completed."}


@app.post("/peft_finetune")
def api_peft_finetune():
    # peft_fine_tuning("meta-llama/Llama-3-8b", PEFT_FINETUNE_PATH)
    return {"status": "PEFT fine-tuning completed."}


@app.post("/rlhf_train")
def api_rlhf_train():
    # rlhf_reward_model_training("meta-llama/Llama-3-8b", RLHF_PREFERENCES_PATH)
    return {"status": "RLHF reward model training completed."}

def main():
    # parser = argparse.ArgumentParser(description="Finance LLM fine-tuning, RAG, and web app")
    # parser.add_argument("--generate_datasets", action="store_true", help="Generate example finance datasets")
    # parser.add_argument("--basic_ft", action="store_true", help="Run basic fine-tuning")
    # parser.add_argument("--peft_ft", action="store_true", help="Run PEFT fine-tuning")
    # parser.add_argument("--rlhf_train", action="store_true", help="Run RLHF reward model training")
    # parser.add_argument("--start_web", action="store_true", help="Start FastAPI web server")
    # parser.add_argument("--model_path", type=str, default="./basic_finetune_results", help="Path to fine-tuned model")
    # args = parser.parse_args()

    # global rag_client, model, tokenizer

    # MODEL_NAME = "meta-llama/Llama-3-8b"

    # if args.generate_datasets:
    #     generate_finance_datasets()

    # if args.basic_ft:
    #     basic_fine_tuning(MODEL_NAME, "datasets/basic_finetune.csv")

    # if args.peft_ft:
    #     peft_fine_tuning(MODEL_NAME, "datasets/peft_finetune.csv")

    # if args.rlhf_train:
    #     rlhf_reward_model_training(MODEL_NAME, "datasets/rlhf_preferences.csv")

    # if args.start_web:
    #     print("Loading model and tokenizer for inference...")
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    #     model = AutoModelForCausalLM.from_pretrained(args.model_path)
    #     model.eval()
    #     rag_client = MilvusRAG(db_path="milvus_rag_db.db")

    #     # Insert example finance documents if Milvus empty
    #     if rag_client.client.count_entities(rag_client.collection_name) == 0:
    #         example_docs = [
    #             "Stocks represent ownership in companies.",
    #             "Compound interest is interest on principal plus accumulated interest.",
    #             "Diversification reduces investment risk by spreading assets.",
    #             "Bonds are fixed income loans from investors to entities.",
    #             "Inflation reduces purchasing power over time."
    #         ]
    #         rag_client.insert_documents(example_docs)

    #     print("Starting FastAPI web server at http://127.0.0.1:8000")
    #     uvicorn.run(app, host="127.0.0.1", port=8000)
    generate_finance_datasets()

if __name__ == "__main__":
    main()
