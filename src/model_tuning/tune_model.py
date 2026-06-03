from __future__ import annotations

from datasets import Dataset, DatasetDict, load_from_disk

from transformers import TrainingArguments

from trl import SFTTrainer

from unsloth import FastLanguageModel


MODEL_NAME = "unsloth/Qwen3-4B-Instruct"
DATASET_PATH = "artifacts/formatted_dataset"
OUTPUT_DIR = "artifacts/model"

MAX_SEQ_LENGTH = 2048


def load_dataset(path: str):

    dataset = load_from_disk(path)

    if isinstance(dataset, DatasetDict):
        return dataset["train"]

    return dataset


def detect_format(dataset: Dataset) -> str:

    cols = set(dataset.column_names)

    if "messages" in cols:
        return "chat"

    if (
        "instruction" in cols
        and "input" in cols
        and "output" in cols
    ):
        return "instruction"

    if (
        "prompt" in cols
        and "response" in cols
    ):
        return "prompt_response"

    raise ValueError(
        f"Unsupported dataset schema: {cols}"
    )


def build_chat_text(
    example,
    tokenizer,
):

    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def build_instruction_text(
    example,
    tokenizer,
):

    messages = [
        {
            "role": "user",
            "content":
                f"{example['instruction']}\n\n"
                f"{example['input']}"
        },
        {
            "role": "assistant",
            "content":
                str(example["output"])
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def build_prompt_response_text(
    example,
    tokenizer,
):

    messages = [
        {
            "role": "user",
            "content":
                str(example["prompt"])
        },
        {
            "role": "assistant",
            "content":
                str(example["response"])
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def prepare_dataset(
    dataset,
    tokenizer,
):

    dataset_format = detect_format(
        dataset
    )

    if dataset_format == "chat":

        dataset = dataset.map(
            lambda x:
            build_chat_text(
                x,
                tokenizer,
            )
        )

    elif dataset_format == "instruction":

        dataset = dataset.map(
            lambda x:
            build_instruction_text(
                x,
                tokenizer,
            )
        )

    else:

        dataset = dataset.map(
            lambda x:
            build_prompt_response_text(
                x,
                tokenizer,
            )
        )

    return dataset


def load_model():

    model, tokenizer = (
        FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
    )

    model = (
        FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
    )

    return model, tokenizer


def train():

    dataset = load_dataset(
        DATASET_PATH
    )

    model, tokenizer = load_model()

    dataset = prepare_dataset(
        dataset,
        tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_ratio=0.03,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            bf16=True,
            report_to="none",
        ),
    )

    trainer.train()

    model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    train()