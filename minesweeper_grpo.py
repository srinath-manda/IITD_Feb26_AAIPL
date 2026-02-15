import os
import sys
import torch
import json
import random
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import torch.nn as nn

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
MODEL_NAME = "unsloth/gpt-oss-20b-BF16" 
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16

# Generation Params (User Requested)
GEN_MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 0.3
GEN_TOP_P = 0.9
GEN_DO_SAMPLE = True

# Training Params (v2 — improved)
LEARNING_RATE = 5e-5
MAX_STEPS = 500
BATCH_SIZE = 4
GRAD_ACCUM = 4
NUM_GENERATIONS = 8  # More exploration per prompt

# Warm-start from previous training
WARM_START_PATH = "your_finetuned_model/lora_weights.pt"

# Game Params
ROWS = 6
COLS = 6
MINES = 5
SEED = 42

# Directories
OUTPUT_DIR = "outputs"
MODEL_SAVE_DIR = "your_finetuned_model"
INPUT_DIR = "inputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)


# ==========================================
# 2. Manual LoRA Implementation (No peft!)
# ==========================================
class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper for nn.Linear — pure PyTorch, no peft needed."""
    def __init__(self, original_linear: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # LoRA matrices A and B
        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=torch.bfloat16)
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=torch.bfloat16)

        # Initialize: A with small random, B with zeros (so LoRA starts at identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_out = self.original_linear(x)
        lora_out = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype)))
        return original_out + lora_out * self.scaling


def apply_lora_to_model(model, r=16, alpha=32, target_modules=None):
    """Apply LoRA to target linear layers in the model."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_params = []
    replaced = 0

    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Get parent module
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)

                # Replace with LoRA (move to same device as original)
                device = module.weight.device
                lora_layer = LoRALinear(module, r=r, alpha=alpha)
                lora_layer = lora_layer.to(device)
                setattr(parent, parts[-1], lora_layer)

                # Collect trainable params
                lora_params.extend(lora_layer.lora_A.parameters())
                lora_params.extend(lora_layer.lora_B.parameters())
                replaced += 1
                break

    print(f"  Applied LoRA to {replaced} layers (r={r}, alpha={alpha})")
    
    # Freeze everything except LoRA
    for param in model.parameters():
        param.requires_grad = False
    for p in lora_params:
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    return model, lora_params


def save_lora_weights(model, save_dir):
    """Save only the LoRA adapter weights."""
    os.makedirs(save_dir, exist_ok=True)
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A.weight"] = module.lora_A.weight.data.cpu()
            lora_state[f"{name}.lora_B.weight"] = module.lora_B.weight.data.cpu()
    torch.save(lora_state, os.path.join(save_dir, "lora_weights.pt"))
    print(f"  Saved {len(lora_state)} LoRA tensors to {save_dir}/lora_weights.pt")


# ==========================================
# 3. Minesweeper Game Engine
# ==========================================
@dataclass
class MinesweeperGame:
    rows: int
    cols: int
    num_mines: int
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _revealed: Set[Tuple[int, int]] = field(init=False, repr=False, default_factory=set)
    _flagged: Set[Tuple[int, int]] = field(init=False, repr=False, default_factory=set)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        if self.num_mines >= self.rows * self.cols:
            raise ValueError("Too many mines for board size")
        self._rng = random.Random(self.seed)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self._place_mines()
        self._calculate_numbers()

    def _place_mines(self):
        positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        mine_positions = self._rng.sample(positions, self.num_mines)
        for r, c in mine_positions:
            self._board[r][c] = -1

    def _calculate_numbers(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self._board[r][c] == -1: continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if self._board[nr][nc] == -1: count += 1
                self._board[r][c] = count

    def _reveal_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols): return False
        if (row, col) in self._revealed or (row, col) in self._flagged: return False

        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in self._revealed: continue
            self._revealed.add((r, c))

            if self._board[r][c] == -1:
                self._state = "failed"
                return True

            if self._board[r][c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.rows and 0 <= nc < self.cols and 
                            (nr, nc) not in self._revealed and (nr, nc) not in self._flagged):
                            stack.append((nr, nc))
        return True

    def _flag_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols): return False
        if (row, col) in self._revealed: return False
        if (row, col) in self._flagged:
            self._flagged.remove((row, col))
        else:
            self._flagged.add((row, col))
        return True

    def do_action(self, action: dict) -> str:
        if self._state != "ongoing": return "game_over"
        if not isinstance(action, dict): 
            self._state = "failed"; return "invalid_format"
        
        a_type = action.get("type")
        row = action.get("row")
        col = action.get("col")

        if a_type not in ["reveal", "flag"] or row is None or col is None:
            self._state = "failed"; return "invalid_format"
        
        try: row, col = int(row), int(col)
        except: self._state = "failed"; return "invalid_format"

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            self._state = "failed"; return "out_of_bounds"

        if a_type == "reveal":
            if (row, col) in self._revealed: 
                self._state = "failed"; return "already_revealed"
            if (row, col) in self._flagged:
                self._state = "failed"; return "flagged_cell"
            valid = self._reveal_cell(row, col)
        else:
            if (row, col) in self._revealed:
                self._state = "failed"; return "invalid_flag"
            valid = self._flag_cell(row, col)

        if not valid: 
            self._state = "failed"; return "invalid_format"

        self._check_win()
        if self._state == "failed": return "mine"
        if self._state == "success": return "win"
        return "ok"

    def _check_win(self):
        safe_cells = (self.rows * self.cols) - self.num_mines
        if len(self._revealed) == safe_cells:
            self._state = "success"

    def get_visible_board(self) -> List[List[str]]:
        visible = []
        for r in range(self.rows):
            row_vis = []
            for c in range(self.cols):
                if (r, c) in self._flagged: row_vis.append('F')
                elif (r, c) in self._revealed:
                    val = self._board[r][c]
                    row_vis.append('*' if val == -1 else str(val))
                else: row_vis.append('.')
            visible.append(row_vis)
        return visible
    
    def state(self) -> str: return self._state


# ==========================================
# 4. Helpers: Formatting & Parsing
# ==========================================
def format_state_for_llm(game: MinesweeperGame) -> str:
    state = {
        "board": game.get_visible_board(),
        "rows": game.rows,
        "cols": game.cols,
        "mines": game.num_mines,
        "flags_placed": len(game._flagged),
        "cells_revealed": len(game._revealed),
    }
    prompt = f"""You are an expert Minesweeper player. Analyze the board carefully and choose the SAFEST move.

Game state:
{json.dumps(state, indent=2)}

Rules:
- "." = unrevealed, "F" = flagged mine, "0"-"8" = adjacent mine count
- Numbers tell you EXACTLY how many of the 8 neighboring cells are mines
- If a number equals its flagged neighbors, all other hidden neighbors are SAFE to reveal
- If a number equals its hidden+flagged neighbors, all hidden neighbors are MINES — FLAG them
- Use "flag" to mark cells you are CERTAIN are mines
- Use "reveal" only on cells you are CERTAIN are safe
- Hitting a mine = instant death. When unsure, flag rather than reveal.

Output ONLY one JSON action:
{{"type": "reveal", "row": R, "col": C}} or {{"type": "flag", "row": R, "col": C}}

Your action:"""
    return prompt

def parse_llm_action(response: str) -> dict:
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
# 5. Reward Functions (12-Point System)
# ==========================================
def is_deducible(game, r, c, action_type):
    visible = game.get_visible_board()
    rows, cols = game.rows, game.cols
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr==0 and dc==0: continue
            nr, nc = r+dr, c+dc
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            
            cell_char = visible[nr][nc]
            if cell_char in [str(i) for i in range(1, 9)]:
                val = int(cell_char)
                
                n_flags = 0
                n_hidden = 0
                for dr2 in [-1, 0, 1]:
                    for dc2 in [-1, 0, 1]:
                        if dr2==0 and dc2==0: continue
                        nnr, nnc = nr+dr2, nc+dc2
                        if 0 <= nnr < rows and 0 <= nnc < cols:
                            if visible[nnr][nnc] == 'F': n_flags += 1
                            if visible[nnr][nnc] == '.': n_hidden += 1
                            
                if action_type == "reveal":
                    if n_flags == val: return True
                elif action_type == "flag":
                    if (n_flags + n_hidden) == val: return True
    return False

def valid_json_reward(completions, **kwargs):
    scores = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else c
        if parse_llm_action(text): scores.append(2.0)
        else: scores.append(-8.0)
    return scores

def gameplay_scores(completions, **kwargs):
    scores = []
    seeds = kwargs.get("seed", [])
    move_histories = kwargs.get("move_history", [])

    for idx, completion in enumerate(completions):
        response = completion[0]["content"] if isinstance(completion, list) else completion
        action = parse_llm_action(response)

        if action is None:
            scores.append(-10.0)
            continue

        if idx >= len(seeds): 
            scores.append(0.0); continue

        seed = seeds[idx]
        hist_raw = move_histories[idx]
        history = json.loads(hist_raw) if isinstance(hist_raw, str) else hist_raw

        game = MinesweeperGame(rows=ROWS, cols=COLS, num_mines=MINES, seed=seed)
        for prev in history: game.do_action(prev)
        
        a_type = action["type"]
        try:
            r, c = int(action["row"]), int(action["col"])
        except (ValueError, TypeError):
            scores.append(-10.0)
            continue

        if not (0 <= r < game.rows and 0 <= c < game.cols):
            scores.append(-15.0)
            continue
        
        cell_val = game._board[r][c]
        is_revealed = (r, c) in game._revealed
        is_flagged = (r, c) in game._flagged
        deduced = is_deducible(game, r, c, a_type)

        reward = 0.0
        
        if a_type == "flag":
            if is_flagged or is_revealed:
                reward = -8.0
            elif len(game._flagged) >= game.num_mines:
                reward = -10.0
            else:
                if cell_val == -1:
                    reward = 25.0 if deduced else 20.0  # Strong flag reward
                else:
                    reward = -15.0  # Penalize wrong flags harder
        
        elif a_type == "reveal":
            if is_revealed:
                reward = -12.0
            elif is_flagged:
                reward = -12.0
            else:
                if cell_val == -1: 
                    reward = -50.0  # Much harsher mine penalty
                else:
                    if deduced: reward = 20.0  # Bigger deduction bonus
                    else: reward = 10.0

        res = game.do_action(action)
        if res == "win":
            reward += 100.0
        
        scores.append(reward)

    return scores


# ==========================================
# 6. Data Generation
# ==========================================
def generate_dataset(num_samples=2000):
    print(f"Generating {num_samples} training samples...")
    items = []
    
    for i in range(num_samples):
        seed = random.randint(0, 1000000)
        game = MinesweeperGame(rows=ROWS, cols=COLS, num_mines=MINES, seed=seed)
        
        # More mid-game states: 80% get 1-10 moves, 20% fresh starts
        if random.random() < 0.8:
            target_moves = random.randint(1, 10)
        else:
            target_moves = 0
        
        history = []
        valid_setup = True
        for _ in range(target_moves):
            vis = game.get_visible_board()
            opts = [(r,c) for r in range(ROWS) for c in range(COLS) if vis[r][c] == '.']
            if not opts: break
            
            safe_opts = [o for o in opts if game._board[o[0]][o[1]] != -1]
            if not safe_opts: 
                valid_setup = False; break
            
            mr, mc = random.choice(safe_opts)
            act = {"type": "reveal", "row": mr, "col": mc}
            res = game.do_action(act)
            if res not in ["ok", "win"]: 
                valid_setup = False; break
            history.append(act)
            if res == "win": break

        if valid_setup and game.state() == "ongoing":
            items.append({
                "prompt": format_state_for_llm(game),
                "seed": seed,
                "move_history": json.dumps(history)
            })
            
    return Dataset.from_list(items)


# ==========================================
# 7. Main Training Pipeline
# ==========================================
def main():
    print("=" * 60)
    print("Minesweeper GRPO Training (No unsloth, No peft)")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load tokenizer
    print(">>> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(">>> Loading model in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply manual LoRA
    print(">>> Applying LoRA (pure PyTorch)...")
    model, lora_params = apply_lora_to_model(
        model, r=LORA_RANK, alpha=LORA_RANK * 2
    )

    # Warm start: load previous LoRA weights if available
    if os.path.exists(WARM_START_PATH):
        print(f">>> Warm-starting from {WARM_START_PATH}")
        prev_state = torch.load(WARM_START_PATH, map_location="cpu", weights_only=True)
        model_state = model.state_dict()
        loaded = 0
        for k, v in prev_state.items():
            if k in model_state:
                model_state[k].copy_(v.to(model_state[k].device))
                loaded += 1
        print(f"  Loaded {loaded}/{len(prev_state)} previous LoRA tensors")
    else:
        print(">>> Training from scratch (no warm start found)")

    # CRITICAL: enable input require grads for gradient checkpointing with LoRA
    model.enable_input_require_grads()

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Generate Data
    dataset = generate_dataset(num_samples=2000)

    # GRPO Config
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=600,
        max_completion_length=GEN_MAX_NEW_TOKENS,
        logging_steps=10,
        save_steps=100,
        report_to="none",
        use_vllm=False,
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[valid_json_reward, gameplay_scores],
        args=training_args,
        train_dataset=dataset,
    )

    print(">>> Starting GRPO Training...")
    trainer.train()

    print(f">>> Saving LoRA weights to {MODEL_SAVE_DIR}...")
    save_lora_weights(model, MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print(">>> SUCCESS!")


if __name__ == "__main__":
    main()
