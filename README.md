# 🎮 Minesweeper RL Agent — Gaming the Models Hackathon

## Team 69

### Approach: GRPO (Group Relative Policy Optimization) with LoRA

**Model**: `unsloth/gpt-oss-20b-BF16` (20B parameters)
**Hardware**: AMD MI300X (274.5 GB VRAM)
**Training**: 500 GRPO steps, ~4.6 hours

### Training Results

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| Total Reward | -14.4 | +7.8 | **+22 pts** |
| Valid JSON % | ~60% | **100%** | ✅ Perfect |
| Gameplay Score | -9.6 | +6.9 | **+16 pts** |

### Architecture

```
gpt-oss-20b-BF16 (frozen) + LoRA (r=16, alpha=32)
├── 20.9B total parameters
├── 7.96M trainable LoRA parameters (0.04%)
├── 96 LoRA target layers (q/k/v/o/gate/up/down_proj)
└── GRPO with 12-point reward system
```

### Reward System (12-Point Scoring)
- **+15** Flag a mine correctly
- **+10/+15** Reveal safe cell (with logic deduction bonus)
- **+100** Win the game
- **-25** Reveal mine (death)
- **-10** Flag safe cell / Invalid move
- **-15** Out of bounds
- **-12** Already revealed cell
- **-8** Flag already flagged cell

### Files

| File | Description |
|------|-------------|
| `minesweeper_grpo.py` | Main GRPO training script |
| `agents/minesweeper_agent.py` | Inference agent & evaluation |
| `notebooks/evaluation.py` | Demo notebook (copy cells to Jupyter) |
| `your_finetuned_model/` | Trained LoRA weights |
| `outputs/` | Training logs |
| `inputs/` | Generated training data |

### How to Run

```bash
# Train (already completed)
python minesweeper_grpo.py

# Evaluate trained model
python agents/minesweeper_agent.py

# Or use notebooks/evaluation.py in Jupyter
```

### Key Innovation: Logic Deduction Bonus
Our reward system includes a **logic deduction detector** that gives extra reward (+15 vs +10) when the model makes a move that can be logically deduced from the current board state, encouraging strategic play rather than random guessing.
