"""
Minesweeper RL Agent — Inference & Evaluation
Loads the GRPO-trained LoRA model and plays Minesweeper games.
"""
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
# Define potential paths for LoRA weights - robust for different environments
LORA_WEIGHTS_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "minesweeper_model", "lora_weights.pt"),
    "/workspace/your_finetuned_model/lora_weights.pt",
    "lora_weights.pt",
    "../lora_weights.pt"
]
MAX_SEQ_LENGTH = 1024
ROWS = 8
COLS = 8
MINES = 10


# ==========================================
# LoRA Implementation (must match training)
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
    """Load saved LoRA weights into the model."""
    lora_path = None
    if isinstance(paths, str):
        paths = [paths]
        
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
# Minesweeper Game Engine
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

    def _reveal_cell(self, row, col):
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

    def _flag_cell(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols): return False
        if (row, col) in self._revealed: return False
        if (row, col) in self._flagged:
            self._flagged.remove((row, col))
        else:
            self._flagged.add((row, col))
        return True

    def do_action(self, action):
        if self._state != "ongoing": return "game_over"
        if not isinstance(action, dict):
            self._state = "failed"; return "invalid_format"
        a_type = action.get("type")
        row, col = action.get("row"), action.get("col")
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
            self._reveal_cell(row, col)
        else:
            if (row, col) in self._revealed:
                self._state = "failed"; return "invalid_flag"
            self._flag_cell(row, col)
        self._check_win()
        if self._state == "failed": return "mine"
        if self._state == "success": return "win"
        return "ok"

    def _check_win(self):
        safe_cells = (self.rows * self.cols) - self.num_mines
        if len(self._revealed) == safe_cells:
            self._state = "success"

    def get_visible_board(self):
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

    def state(self): return self._state

    def display(self):
        vis = self.get_visible_board()
        print("   " + " ".join([str(c) for c in range(self.cols)]))
        print("  +" + "--" * self.cols + "+")
        for r, row in enumerate(vis):
            print(f" {r}| " + " ".join(row) + " |")
        print("  +" + "--" * self.cols + "+")
        print(f"  State: {self._state} | Revealed: {len(self._revealed)} | Flagged: {len(self._flagged)}")


# ==========================================
# Agent
# ==========================================
def format_prompt_from_dict(state: Dict[str, Any]) -> str:
    """Format prompt from a dictionary state (used by evaluation)."""
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


def format_prompt(game: MinesweeperGame) -> str:
    """Format prompt from a MinesweeperGame object (used by internal loop)."""
    state = {
        "board": game.get_visible_board(),
        "rows": game.rows, "cols": game.cols,
        "mines": game.num_mines,
        "flags_placed": len(game._flagged),
        "cells_revealed": len(game._revealed),
    }
    return format_prompt_from_dict(state)


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


class MinesweeperPlayer:
    """
    Player class for the evaluation harness.
    """
    def __init__(self, model_override=None):
        print(f"Initializing MinesweeperPlayer...")
        
        # Load model slightly differently depending on if we are running in eval or standalone
        # logic here is simplified for the demo
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
        """
        Play a single action given the game state.
        
        Args:
            game_state: Dict containing 'board', 'rows', 'cols', etc.
            **kwargs: Generation arguments (e.g. max_new_tokens, temperature)
            
        Returns:
            (action_dict, tokens_generated, generation_time)
        """
        prompt = format_prompt_from_dict(game_state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Merge default kwargs with provided ones
        gen_kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)
        # Remove custom args if they exist to avoid errors in model.generate
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


class MinesweeperAgent:
    """
    Wrapper for internal testing using the MinesweeperGame class.
    """
    def __init__(self, model, tokenizer, device="cuda"):
        # This wrapper now just uses the Player class internally or shares logic
        # For backward compatibility with the existing main() loop
        self.player = MinesweeperPlayer(model_override=model)
        self.player.tokenizer = tokenizer # Ensure tokenizer is shared
        
    def get_action(self, game, temperature=0.3, top_p=0.9):
        # Convert game object to state dict
        state = {
            "board": game.get_visible_board(),
            "rows": game.rows, "cols": game.cols,
            "mines": game.num_mines,
            "flags_placed": len(game._flagged),
            "cells_revealed": len(game._revealed),
        }
        action, _, _ = self.player.play_action(state, temperature=temperature, top_p=top_p)
        return action, "raw response not captured in this wrapper shim"

    def play_game(self, seed=None, max_moves=50, verbose=True):
        game = MinesweeperGame(rows=ROWS, cols=COLS, num_mines=MINES, seed=seed)
        moves = 0
        total_reward = 0

        if verbose:
            print(f"\n{'='*50}")
            print(f"  Game (seed={seed})")
            print(f"{'='*50}")
            game.display()

        while game.state() == "ongoing" and moves < max_moves:
            action, _ = self.get_action(game)

            if action is None:
                if verbose: print(f"  Move {moves+1}: INVALID (no valid JSON)")
                total_reward -= 10
                break

            result = game.do_action(action)
            moves += 1

            if result == "ok":
                reward = 10
            elif result == "win":
                reward = 100
            elif result == "mine":
                reward = -25
            else:
                reward = -10

            total_reward += reward

            if verbose:
                print(f"\n  Move {moves}: {json.dumps(action)} → {result} (reward: {reward:+.0f})")
                game.display()

        if verbose:
            print(f"\n  Result: {game.state()} | Moves: {moves} | Total Reward: {total_reward:+.0f}")

        return {
            "state": game.state(),
            "moves": moves,
            "reward": total_reward,
            "seed": seed,
        }


def evaluate(agent, num_games=20, verbose=False):
    """Run evaluation games and report statistics."""
    results = []
    for i in range(num_games):
        seed = 10000 + i
        result = agent.play_game(seed=seed, verbose=verbose)
        results.append(result)

    wins = sum(1 for r in results if r["state"] == "success")
    losses = sum(1 for r in results if r["state"] == "failed")
    avg_reward = np.mean([r["reward"] for r in results])
    avg_moves = np.mean([r["moves"] for r in results])

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS ({num_games} games)")
    print(f"{'='*50}")
    print(f"  Wins:        {wins}/{num_games} ({100*wins/num_games:.1f}%)")
    print(f"  Losses:      {losses}/{num_games}")
    print(f"  Avg Reward:  {avg_reward:+.1f}")
    print(f"  Avg Moves:   {avg_moves:.1f}")
    print(f"{'='*50}")

    return results


# ==========================================
# Main
# ==========================================
def main():
    print("=" * 60)
    print("  Minesweeper RL Agent — Inference & Evaluation")
    print("=" * 60)

    # The Player class handles loading now
    player = MinesweeperPlayer()
    
    # Create the wrapper for the existing evaluation loop
    agent = MinesweeperAgent(player.model, player.tokenizer)

    # Demo: Play 3 games with verbose output
    print("\n" + "=" * 60)
    print("  DEMO GAMES (verbose)")
    print("=" * 60)
    for seed in [42, 123, 456]:
        agent.play_game(seed=seed, verbose=True)

    # Evaluation: 20 games
    evaluate(agent, num_games=20, verbose=False)


if __name__ == "__main__":
    main()
