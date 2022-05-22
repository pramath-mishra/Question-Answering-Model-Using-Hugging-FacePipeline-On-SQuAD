import os

import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import pipeline

# loading SQUAD V2 dataset
squad = load_dataset("squad_v2")
print(f"SQUAD V2 dataset loaded...\n -sample: {squad}")

# creating model pipeline object
qa_model = pipeline('question-answering', model='SQuAD_model', tokenizer='SQuAD_model')
print("model pipeline object created...")

# inference
for i, data in enumerate(squad["validation"]):
    result = qa_model(question=data["question"], context=data["context"])
    print(f"\nsample: {i+1}\n", file=open("./result_sample.txt", "a"))
    print(f"Question: {data['question']}", file=open("./result_sample.txt", "a"))
    print(f"Context: {data['context']}", file=open("./result_sample.txt", "a"))
    print(f"Answer: {result['answer']}", file=open("./result_sample.txt", "a"))
    print(f"Score: {str(round(result['score'], 4)*100)+'%'}", file=open("./result_sample.txt", "a"))
    if i == 10:
        break
