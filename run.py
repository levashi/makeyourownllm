import torch
import os
import json
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str, help="The prompt to use for inference. If not specified, the script will ask for it.")
parser.add_argument("--device", type=torch.device, default="cuda", help="Device to use for inference")
parser.add_argument("--model-path", "-m", type=str, help="Path to the model directory.")
parser.add_argument("--config-path", type=str, help="Path to the config file of the inference. If not specified, we will use default parameters.")
parser.add_argument("--tokenizer-path", "-t", type=str, default="tokenizer.json", help="Path to the tokenizer file.")
parser.add_argument("-use-auto-tokenizer", action="store_true", help="Use the auto tokenizer.")
parser.add_argument("--output", "-o", type=str, help="If specified, the generated text will be saved to the specified file.")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

if args.config_path:
    with open(args.config_path, "r") as f:
        config = json.load(f)
else:
    config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 50,
        "num_beams": 1,
        "max_new_tokens": 50,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "early_stopping": False,
        "seed": 42
    }

if args.device == "cuda" and not torch.cuda.is_available():
    logging.info("CUDA is not available. Using CPU.")
    args.device = "cpu"

if not args.model_path or not os.path.exists(os.path.join(args.model_path)):
    raise ValueError("Please provide a valid model path with -m or --model-path")

if (not args.tokenizer_path or not os.path.exists(os.path.join(args.model_path, args.tokenizer_path))) and not args.use_auto_tokenizer:
    raise ValueError("Please provide a valid tokenizer path with --tokenizer-path or -t. Also, you can use the flag --use-auto-tokenizer to use the auto tokenizer (recommended if you don't have a tokenizer).")

device = args.device
model_path = args.path
prompt = args.prompt

if not prompt:
    prompt = str(input("Enter the prompt: "))

if args.use_auto_tokenizer:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
else:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(model_path, args.tokenizer_path))

model = torch.load(os.path.join(model_path), map_location=device)

with torch.no_grad():
    input_ids = tokenizer.encode(prompt).ids
    if len(input_ids) > tokenizer.model_max_length:
        input_ids = input_ids[-tokenizer.model_max_length:]
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens=config["max_new_tokens"], temperature=config["temperature"], top_p=config["top_p"], top_k=config["top_k"], num_beams=config["num_beams"], do_sample=config["do_sample"], repetition_penalty=config["repetition_penalty"], length_penalty=config["length_penalty"], early_stopping=config["early_stopping"], seed=config["seed"])
    generated = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    
logging.info(f"Generated: {generated}")
if args.output:
    with open(args.output, "w") as f:
        f.write(generated)
    logging.info(f"Generated text saved to {args.output}")