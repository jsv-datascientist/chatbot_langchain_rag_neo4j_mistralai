import CustomLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import os

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, BitsAndBytesConfig


def load_model(training_config, load_base_model=False):
    model_load_path = training_config["model"]["pretrained_name"]

    print(f"Loading default model : {model_load_path}")
    model = AutoModelForCausalLM.from_pretrained(model_load_path)
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)

    print("Copying model to device")

    device_count = torch.cuda.device_count()
    if device_count > 0:
        print("Select GPU Device")
        device = torch.device("cuda")
    else:
        print("Select CPU device")
        device = torch.device("cpu")
    # model.to_device()

    print("Copying to local device finished...")

    if "model_name" not in training_config:
        model_name = model_load_path
    else:
        model_name = training_config["model_name"]

    return model, tokenizer, device, model_name


def main():
    model_load_path = "./saved_models/Mistral-7B-Instruct-v0.2/"

    # List all files and directories
    # files_and_directories = os.listdir(model_load_path)
    # files = [f for f in files_and_directories]
    # print(files)
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_load_path, local_files_only=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)

    # model.save_pretrained("./saved_moodels/mistralai/")
    # tokenizer.save_pretrained("./saved_moodels/mistralai/")
    # output = CustomLLM(model, tokenizer)

    return ""


if __name__ == "__main__":
    model_name = "drive/MyDrive/branded-llm-research/saved_models/hf/4bit/mistralai/Mistral-7B-Instruct-v0.2"
    dataset_path = "drive/MyDrive/branded-llm-research/llm_jsonl_data/puracy_identity_data.jsonl"

    use_hf = False
    # Load the model
    training_config = {
        "model": {
            "pretrained_name": model_name,
            "max_length": 2048
        },
        "datasets": {
            "use_hf": use_hf,
            "path": dataset_path
        },
        "verbose": True
    }

    result = load_model(training_config)
    print(result)
