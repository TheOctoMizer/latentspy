import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import argparse
import os
import random

def generate_sample(model, tokenizer, prompt, device, max_length=50):
    """Generate a single sample with fixed seeding for total fairness."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True).replace("\n", " ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LatentSpy model comparison tool")
    parser.add_argument("--path", type=str, default="models", help="Path to a model directory (to test one) or a parent directory (to compare many)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    
    found_models = {}
    
    if os.path.exists(os.path.join(args.path, "config.json")):
        found_models[os.path.basename(args.path.rstrip("/"))] = args.path
    elif os.path.isdir(args.path):
        for sub in sorted(os.listdir(args.path)):
            sub_path = os.path.join(args.path, sub)
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "config.json")):
                found_models[sub] = sub_path

    if not found_models:
        print(f"Error: No valid models found at '{args.path}'. (Looking for directories with 'config.json')")
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print("\n" + "="*80)
        print(f"{'LATENTSPY MODEL COMPARISON':^80}")
        print("="*80)
        
        loaded_models = {}
        for name, path in found_models.items():
            print(f"Loading [{name}]...")
            loaded_models[name] = GPT2LMHeadModel.from_pretrained(path).to(device)
            loaded_models[name].eval()

        prompts = [
            "Once upon a time, there was a little",
            "The brave knight decided to",
            "In a far away kingdom, a",
            "One day, a curious bird found",
            "Deep in the forest, the"
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"\nPROMPT {i}: \"{prompt}\"")
            print("-" * 40)
            
            for name, model in loaded_models.items():
                output = generate_sample(model, tokenizer, prompt, device)
                print(f"[{name:<20}] {output}")
        
        print("\n" + "="*80)
