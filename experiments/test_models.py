import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import argparse
import os

def run_acid_test(model_path, device="cpu"):
    print(f"=== ACID TEST: {model_path} ===")
    
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        return

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()

    prompts = [
        "Once upon a time, there was a little",
        "The brave knight decided to",
        "In a far away kingdom, a",
        "One day, a curious bird found",
        "Deep in the forest, the"
    ]

    print("\n--- Generating Samples ---\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_length=50,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(f"Output: {text}\n" + "-"*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LatentSpy ACID Test for TinyStories Models")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model directory")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    run_acid_test(args.model, device=device)
