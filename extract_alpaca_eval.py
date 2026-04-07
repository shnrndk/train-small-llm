import config
import data_utils as du
import pandas as pd

def main():
    print("Loading Alpaca dataset with original seeds...")
    
    # 1. Grab the exact same config you used for training
    model_args, data_args, lora_args, train_args = config.get_config_classes(training_type="sft")

    # 2. Re-create the exact same split
    train_set, val_set = du.prepare_alpaca_data(
        dataset_name=data_args.dataset_name,
        process_func=du.alpaca_row_to_text_train,
        validation_size=data_args.validation_size,
        seed=data_args.dataset_ops_seed,
    )

    print(f"Validation dataset size to extract: {len(val_set)}")

    # 3. Convert the HuggingFace dataset to a Pandas DataFrame
    df = val_set.to_pandas()

    # 4. Save to CSV and JSON
    csv_path = "alpaca_eval_set.csv"
    json_path = "alpaca_eval_set.json"
    
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=4)
    
    print(f"Success! Saved evaluation sets to:")
    print(f" - {csv_path}")
    print(f" - {json_path}")

if __name__ == "__main__":
    main()