# Minesweeper Evaluation Script for Notebooks
# COPY-PASTE THIS ENTIRE CELL INTO YOUR NOTEBOOK (Colab/Kaggle)
# Ensure 'lora_weights.pt' is in the same directory or update LORA_WEIGHTS_PATH

import os
import sys
import torch
import json
import random
import re
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

# ==========================================
# Configuration
# ==========================================
BASE_MODEL = "unsloth/gpt-oss-20b-BF16"
# Define potential paths for LoRA weights
LORA_WEIGHTS_PATHS = [
    "lora_weights.pt",
    "/workspace/your_finetuned_model/lora_weights.pt",  # Updated based on previous logs
    "../your_finetuned_model/lora_weights.pt",
    "../lora_weights.pt"
]
MAX_SEQ_LENGTH = 1024
ROWS = 8
COLS = 8
MINES = 10

# ==========================================
# LoRA Implementation
# ==========================================
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=torch.bfloat16)
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=torch.bfloat16)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_out = self.original_linear(x)
        lora_out = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype)))
        return original_out + lora_out * self.scaling

def apply_lora_to_model(model, r=16, alpha=32):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                device = module.weight.device
                lora_layer = LoRALinear(module, r=r, alpha=alpha)
                lora_layer = lora_layer.to(device)
                setattr(parent, parts[-1], lora_layer)
                break
    return model

def load_lora_weights(model, paths):
    lora_path = None
    for path in paths:
        if os.path.exists(path):
            lora_path = path
            break
    
    if lora_path is None:
        print(f"Warning: LoRA weights not found in any of these locations: {paths}")
        print("Using base model (which will likely fail).")
        return model

    print(f"Loading LoRA weights from: {lora_path}")
    state_dict = torch.load(lora_path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    loaded = 0
    for key, val in state_dict.items():
        if key in model_state:
            model_state[key].copy_(val.to(model_state[key].device))
            loaded += 1
    print(f"  Loaded {loaded}/{len(state_dict)} LoRA tensors")
    return model

# ==========================================
# Game Logic
# ==========================================
def format_prompt_from_dict(state: Dict[str, Any]) -> str:
    return f"""You are playing Minesweeper. Analyze the game state and output your next move.

Game state:
{json.dumps(state, indent=2)}

Legend:
- "." = unrevealed cell
- "F" = flagged cell (suspected mine)
- "0"-"8" = number of adjacent mines
- "*" = revealed mine (game over)

Output your next action as JSON:
{{"type": "reveal", "row": <row_index>, "col": <col_index>}}
or
{{"type": "flag", "row": <row_index>, "col": <col_index>}}

Your action:"""

def parse_action(response):
    best = None
    for match in re.finditer(r'\{[^{}]*\}', response):
        try:
            action = json.loads(match.group())
            if ("type" in action and "row" in action and "col" in action
                    and action["type"] in ["reveal", "flag"]):
                best = action
        except: continue
    return best

# ==========================================
# Player Agent
# ==========================================
class MinesweeperPlayer:
    def __init__(self, model_override=None):
        print(f"Initializing MinesweeperPlayer...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_override:
            self.model = model_override
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        else:
            print(f"Loading base model: {BASE_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            
            print("Applying LoRA adapters...")
            self.model = apply_lora_to_model(self.model, r=16, alpha=32)
            
            print(f"Loading trained weights...")
            self.model = load_lora_weights(self.model, LORA_WEIGHTS_PATHS)
            self.model.eval()

    @torch.no_grad()
    def play_action(self, game_state: Dict[str, Any], **kwargs) -> Tuple[Optional[Dict], Optional[int], Optional[float]]:
        prompt = format_prompt_from_dict(game_state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)
        gen_kwargs.pop("tgps_show", None) 

        start_time = time.time()
        outputs = self.model.generate(**inputs, **gen_kwargs)
        end_time = time.time()
        
        gen_time = end_time - start_time
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        token_count = len(new_tokens)
        
        action = parse_action(response)
        return action, token_count, gen_time

# ==========================================
# Evaluation Logic
# ==========================================
SAMPLE_GAME_STATE = {
    "board": [
        ["1", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
    ],
    "rows": 8, "cols": 8, "mines": 10,
    "flags_placed": 0, "cells_revealed": 1,
}

def pretty_board(game_state: dict) -> str:
    board = game_state["board"]
    cols = len(board[0])
    header = "     " + "  ".join(f"{i:2d}" for i in range(cols))
    sep = "    " + "-" * (cols * 4 + 1)
    lines = [header, sep]
    for r, row in enumerate(board):
        cells = "  ".join(f" {c}" for c in row)
        lines.append(f" {r:2d} | {cells}")
    return "\n".join(lines)

def run_evaluation():
    print("="*60)
    print("Minesweeper Evaluation Demo")
    print("="*60)
    
    # 1. Load Player
    try:
        player = MinesweeperPlayer()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Setup Game State
    game_state = SAMPLE_GAME_STATE
    print(f"\nBoard State:")
    print(pretty_board(game_state))
    
    # 3. Run Inference
    print("\nRunning inference...")
    start = time.time()
    action, tokens, gen_time = player.play_action(game_state)
    elapsed = time.time() - start
    
    # 4. Results
    print(f"\nAction Selected: {action}")
    print(f"Time: {elapsed:.2f}s")
    if tokens:
        print(f"Tokens: {tokens} ({tokens/gen_time:.1f} tok/s)")
    
    if action:
        print("\nSUCCESS! The agent produced a valid action.")
    else:
        print("\nFAILED. The agent did not produce a valid JSON action.")

if __name__ == "__main__":
    run_evaluation()
