import numpy as np
from datasets import load_dataset


# Alpaca prompt template
ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task{input_block}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_section}"
    "### Response:\n"
)


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """
    Format an Alpaca-style prompt given an instruction and optional input.
    When input_text is non-empty, it is included under a '### Input:' section.
    """
    if input_text.strip():
        input_block = ", paired with an input that provides further context"
        input_section = f"### Input:\n{input_text}\n\n"
    else:
        input_block = ""
        input_section = ""

    return ALPACA_PROMPT_TEMPLATE.format(
        input_block=input_block,
        instruction=instruction,
        input_section=input_section,
    )


def alpaca_row_to_text_train(sample):
    """
    Convert a single Alpaca dataset row into a full training string for SFTTrainer.
    The response is appended directly after '### Response:' so the model learns
    to complete the prompt. SFTTrainer receives this under the 'text' key.

    Alpaca schema:
        instruction (str): The task instruction.
        input      (str): Optional additional context (often empty string).
        output     (str): The expected response.
    """
    prompt = format_alpaca_prompt(sample["instruction"], sample["input"])
    # Append the target response so the trainer has the correct labels
    full_text = prompt + sample["output"]
    return {"text": full_text}


def alpaca_row_to_prompt_eval(sample):
    """
    Format an Alpaca sample as a plain prompt string for inference.
    Does NOT include the response (used for generation).
    """
    prompt = format_alpaca_prompt(sample["instruction"], sample["input"])
    return {"text": prompt}


def prepare_alpaca_data(
    dataset_name: str,
    process_func: callable,
    validation_size: float,
    seed: int = 42,
):
    """
    Load the Alpaca dataset from the Hub, apply the message-formatting function,
    and return a train/validation split.

    Args:
        dataset_name    (str):      HuggingFace dataset name, e.g. 'tatsu-lab/alpaca'.
        process_func    (callable): Row-level mapping function (train or eval variant).
        validation_size (float):    Fraction of data to hold out for validation.
        seed            (int):      Seed for the train/validation split.

    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, eval_dataset)
    """
    rng = np.random.default_rng(seed)

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    print(f"Total samples loaded: {len(dataset)}")

    # Alpaca has a small number of rows where 'output' is empty — drop them.
    before = len(dataset)
    dataset = dataset.filter(lambda x: len(x["output"].strip()) > 0)
    dropped = before - len(dataset)
    if dropped:
        print(f"Dropped {dropped} samples with empty output fields.")

    processed_dataset = dataset.map(
        process_func,
        batched=False,
        remove_columns=dataset.column_names,
    )

    split = processed_dataset.train_test_split(test_size=validation_size, seed=seed)
    train_dataset = split["train"]
    eval_dataset  = split["test"]

    print(f"Train dataset size:      {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

    return train_dataset, eval_dataset