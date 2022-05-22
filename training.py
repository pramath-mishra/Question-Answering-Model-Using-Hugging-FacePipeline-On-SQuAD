import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

from torch import cuda
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

# availability of GPU
device = 'cuda' if cuda.is_available() else 'cpu'
print(f"Device: {device}")

# loading SQUAD V2 dataset
squad = load_dataset("squad_v2")
print(f"SQUAD V2 dataset loaded...\n -sample: {squad}")

# loading tokenizer & model
model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForQuestionAnswering.from_pretrained(model)
model.to(device)
print("distilbert tokenizer & model loaded...")

# tokenizer parameters
max_length = 384
doc_stride = 128


# preparing train features
def prepare_train_features(examples, pad_on_right=True):
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    return tokenized_examples


tokenized_datasets = squad.map(prepare_train_features, batched=True, remove_columns=squad["train"].column_names)
print("train features prepared...")

# updating training arguments
args = TrainingArguments(
    f"squad-test",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)
print("training arguments updated...")

# create trainer object
data_collator = default_data_collator
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
print("trainer object created...")

# model training
trainer.train()
print("model training done...")

# saving the fine-tuned model
model_path = "SQuAD_model"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print("fine-tuned model & tokenizer saved...")
