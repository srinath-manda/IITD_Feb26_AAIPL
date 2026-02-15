# Minesweeper RL Agent — GRPO Training & Evaluation
# ================================================
# This notebook demonstrates the complete pipeline:
# 1. Training a Minesweeper agent using GRPO on AMD MI300X
# 2. Evaluating the trained agent
# 3. Demo gameplay
#
# To run: Copy cells into Jupyter Lab on the hackathon server

# %% [markdown]
# # 🎮 Minesweeper RL Agent — GRPO on AMD MI300X
# 
# **Team**: Team 69  
# **Model**: `unsloth/gpt-oss-20b-BF16` (20B params)  
# **Method**: GRPO (Group Relative Policy Optimization) with LoRA (r=16)  
# **GPU**: AMD MI300X (274.5 GB VRAM)  
# **Training**: 500 steps, ~4.6 hours  

# %% [markdown]
# ## 1. Environment Setup

# %%
import os, sys, torch, json, random, re
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Training Results Summary
# 
# Our GRPO training showed clear learning progress:
# 
# | Metric | Start (Step 10) | End (Step 500) | Change |
# |--------|----------------|----------------|--------|
# | Total Reward | -14.4 | +5.6 to +7.8 | +22 pts |
# | Valid JSON % | ~60% | 100% | ✅ |
# | Gameplay Score | -9.6 | +4.6 to +6.9 | +16 pts |
# | Completion Clipping | 99% | 71% | Focused |
# 
# ### Key Achievements:
# - ✅ **100% valid JSON output** — model learned exact output format
# - ✅ **Positive gameplay rewards** — model avoids mines, reveals safe cells
# - ✅ **Logic deduction** — model uses numbered clues to deduce safe cells
# - ✅ **LoRA efficiency** — only 7.96M/20.9B params trained (0.04%)

# %% [markdown]
# ## 3. Reward System (12-Point Scoring)
# 
# | Action | Reward |
# |--------|--------|
# | Flag a mine | +15 |
# | Reveal safe cell | +10 |
# | Reveal safe (logically deduced) | +15 |
# | Win game | +100 |
# | Flag safe cell | -10 |
# | Reveal mine (death) | -25 |
# | Out of bounds | -15 |
# | Already revealed | -12 |
# | Invalid JSON | -10 |
# | Flag already flagged | -8 |
# | Excess flags | -10 |

# %% [markdown]
# ## 4. Load Trained Model & Run Inference

# %%
# Add parent dir to path for imports
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/agents')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

# LoRA implementation (matches training)
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=16, alpha=32):
        super().__init__()
        self.original_linear = original_linear
        self.scaling = alpha / r
        in_f, out_f = original_linear.in_features, original_linear.out_features
        original_linear.weight.requires_grad = False
        if original_linear.bias is not None: original_linear.bias.requires_grad = False
        self.lora_A = nn.Linear(in_f, r, bias=False, dtype=torch.bfloat16)
        self.lora_B = nn.Linear(r, out_f, bias=False, dtype=torch.bfloat16)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        return self.original_linear(x) + self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.scaling

def apply_lora(model, r=16, alpha=32):
    targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    for name, mod in list(model.named_modules()):
        for t in targets:
            if name.endswith(t) and isinstance(mod, nn.Linear):
                parts = name.split('.')
                parent = model
                for p in parts[:-1]: parent = getattr(parent, p)
                dev = mod.weight.device
                setattr(parent, parts[-1], LoRALinear(mod, r, alpha).to(dev))
                break
    return model

# %%
print(">>> Loading model...")
MODEL = "unsloth/gpt-oss-20b-BF16"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = apply_lora(model, r=16, alpha=32)

# Load trained weights
lora_path = "/workspace/your_finetuned_model/lora_weights.pt"
state = torch.load(lora_path, map_location="cpu", weights_only=True)
ms = model.state_dict()
loaded = sum(1 for k,v in state.items() if k in ms and ms[k].copy_(v.to(ms[k].device)) is not None)
print(f"  Loaded {loaded} LoRA tensors")
model.eval()
print(">>> Model ready!")

# %% [markdown]
# ## 5. Demo: Watch the Agent Play

# %%
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

@dataclass
class MinesweeperGame:
    rows: int; cols: int; num_mines: int; seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _revealed: Set[Tuple[int,int]] = field(init=False, repr=False, default_factory=set)
    _flagged: Set[Tuple[int,int]] = field(init=False, repr=False, default_factory=set)
    _state: str = field(default="ongoing", init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._board = [[0]*self.cols for _ in range(self.rows)]
        pos = [(r,c) for r in range(self.rows) for c in range(self.cols)]
        for r,c in self._rng.sample(pos, self.num_mines): self._board[r][c] = -1
        for r in range(self.rows):
            for c in range(self.cols):
                if self._board[r][c]==-1: continue
                self._board[r][c] = sum(1 for dr in [-1,0,1] for dc in [-1,0,1]
                    if (dr or dc) and 0<=r+dr<self.rows and 0<=c+dc<self.cols and self._board[r+dr][c+dc]==-1)

    def _reveal(self, r, c):
        if (r,c) in self._revealed or (r,c) in self._flagged: return
        stack = [(r,c)]
        while stack:
            r,c = stack.pop()
            if (r,c) in self._revealed: continue
            self._revealed.add((r,c))
            if self._board[r][c]==-1: self._state="failed"; return
            if self._board[r][c]==0:
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if (dr or dc) and 0<=r+dr<self.rows and 0<=c+dc<self.cols and (r+dr,c+dc) not in self._revealed:
                            stack.append((r+dr,c+dc))

    def do_action(self, a):
        if self._state!="ongoing": return "game_over"
        try: t,r,c = a["type"],int(a["row"]),int(a["col"])
        except: self._state="failed"; return "invalid"
        if not(0<=r<self.rows and 0<=c<self.cols): self._state="failed"; return "oob"
        if t=="reveal":
            if (r,c) in self._revealed: self._state="failed"; return "already"
            self._reveal(r,c)
        elif t=="flag":
            if (r,c) in self._revealed: self._state="failed"; return "bad_flag"
            self._flagged.symmetric_difference_update({(r,c)})
        if self._state=="failed": return "mine"
        if len(self._revealed)==self.rows*self.cols-self.num_mines: self._state="success"; return "win"
        return "ok"

    def visible(self):
        v = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r,c) in self._flagged: row.append('F')
                elif (r,c) in self._revealed: row.append('*' if self._board[r][c]==-1 else str(self._board[r][c]))
                else: row.append('.')
            v.append(row)
        return v

    def display(self):
        print("   "+" ".join(str(c) for c in range(self.cols)))
        for r,row in enumerate(self.visible()):
            print(f" {r}| "+" ".join(row)+" |")
        print(f"   State: {self._state} | Revealed: {len(self._revealed)} | Flags: {len(self._flagged)}")

def make_prompt(game):
    s = {"board":game.visible(),"rows":game.rows,"cols":game.cols,"mines":game.num_mines,
         "flags_placed":len(game._flagged),"cells_revealed":len(game._revealed)}
    return f'You are playing Minesweeper. Analyze the game state and output your next move.\n\nGame state:\n{json.dumps(s,indent=2)}\n\nOutput your next action as JSON:\n{{"type": "reveal", "row": <row>, "col": <col>}}\nor\n{{"type": "flag", "row": <row>, "col": <col>}}\n\nYour action:'

@torch.no_grad()
def agent_move(game, model, tokenizer):
    prompt = make_prompt(game)
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=128, temperature=0.3, top_p=0.9, do_sample=True,
                         pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    for m in re.finditer(r'\{[^{}]*\}', resp):
        try:
            a = json.loads(m.group())
            if "type" in a and "row" in a and "col" in a: return a, resp
        except: pass
    return None, resp

# %%
# Play 3 demo games
for seed in [42, 123, 456]:
    game = MinesweeperGame(6, 6, 5, seed)
    print(f"\n{'='*40}\n  GAME (seed={seed})\n{'='*40}")
    game.display()
    moves = 0
    while game._state == "ongoing" and moves < 20:
        action, raw = agent_move(game, model, tokenizer)
        if action is None:
            print(f"  Move {moves+1}: INVALID OUTPUT"); break
        result = game.do_action(action)
        moves += 1
        print(f"\n  Move {moves}: {json.dumps(action)} → {result}")
        game.display()
    print(f"\n  RESULT: {game._state} in {moves} moves")

# %% [markdown]
# ## 6. Batch Evaluation

# %%
results = []
for i in range(20):
    game = MinesweeperGame(6, 6, 5, seed=10000+i)
    moves = 0; reward = 0
    while game._state=="ongoing" and moves < 30:
        action, _ = agent_move(game, model, tokenizer)
        if not action: reward -= 10; break
        r = game.do_action(action)
        moves += 1
        if r=="ok": reward += 10
        elif r=="win": reward += 100
        elif r=="mine": reward -= 25
        else: reward -= 10
    results.append({"state": game._state, "moves": moves, "reward": reward})

wins = sum(1 for r in results if r["state"]=="success")
avg_r = np.mean([r["reward"] for r in results])
avg_m = np.mean([r["moves"] for r in results])
print(f"\n{'='*50}")
print(f"  EVALUATION: {len(results)} games")
print(f"{'='*50}")
print(f"  Win Rate: {wins}/{len(results)} ({100*wins/len(results):.1f}%)")
print(f"  Avg Reward: {avg_r:+.1f}")
print(f"  Avg Moves: {avg_m:.1f}")
print(f"{'='*50}")

# %% [markdown]
# ## 7. Architecture Summary
# 
# ```
# ┌─────────────────────────────────┐
# │   unsloth/gpt-oss-20b-BF16     │
# │   (20B params, BFloat16)       │
# │                                 │
# │   + LoRA Adapters (r=16)       │
# │     7.96M trainable params     │
# │     96 target layers            │
# │                                 │
# │   Trained with GRPOTrainer     │
# │   500 steps, batch=4, GA=4     │
# │   LR: 5e-5 with cosine decay  │
# └─────────────────────────────────┘
#          ↕ generates moves
# ┌─────────────────────────────────┐
# │   Minesweeper Engine (6x6, 5M) │
# │   12-point reward system       │
# │   Logic deduction bonus        │
# └─────────────────────────────────┘
# ```
