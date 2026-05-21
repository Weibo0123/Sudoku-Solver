# DQN Sudoku Solver — Experiment Summary

## Objective
Train an agent using pure Deep Reinforcement Learning (DQN) to solve 9×9 Sudoku puzzles from scratch, without any supervised learning.

---

## What I Changed

### Step 1: Redesigned SudokuEnv with Answer-Based Rewards
**Problem:** The original reward only checked whether the move was valid by Sudoku rules, without using the known solution.

**Changes:**
- Added `solution` parameter to `__init__()` to store the correct answer
- Replaced `step()` reward logic:
  ```
  Any move (valid by rules)     → +0.1
  Correct move (matches answer) → +1.0
  Wrong move (valid but wrong)  → -0.5
  Puzzle fully solved           → +10.0
  ```
- `create_sudoku()` now returns both `puzzle` and `solution`

**Result:** Dense reward signal — the agent receives immediate feedback on every step, eliminating the sparse reward problem.

---

### Step 2: Replaced MLP with CNN in QNetwork
**Problem:** MLP flattened the board into a 1D vector, losing all spatial structures (rows, columns, boxes).

**Changes:**
- Input changed to one-hot encoded 3D tensor: `(board_size, board_size, board_size)`
- Each cell value encoded as a one-hot vector of length 9 (e.g. digit 3 → `[0,0,1,0,0,0,0,0,0]`)
- Network architecture:
  ```
  Conv2d(9→64) → Conv2d(64→128) → Conv2d(128→256)
  Linear(256×81→512) → Linear(512→256) → Linear(256→output)
  ```
- Added `state_to_tensor()` to handle the board-to-one-hot conversion

**Result:** Loss dropped from 0.5–1.0 to 0.1–0.5. SOLVED frequency increased noticeably. Training became more stable.

---

### Step 3: Curriculum Learning
**Problem:** Starting directly at 9×9 with 20 empty cells, the agent was purely random and never learned anything meaningful.

**Changes:** Defined progressive difficulty stages:
```python
STAGES = [
    {"size": 9, "empty_cells": 3,  "threshold": 0.95},
    {"size": 9, "empty_cells": 6,  "threshold": 0.95},
    {"size": 9, "empty_cells": 10, "threshold": 0.95},
    {"size": 9, "empty_cells": 12, "threshold": 0.9},
    {"size": 9, "empty_cells": 15, "threshold": 0.8},
    {"size": 9, "empty_cells": 20, "threshold": 0.8},
    {"size": 9, "empty_cells": 40, "threshold": None},
]
```
- Evaluated success rate every 100 episodes
- On promotion: reset epsilon to 0.5 to encourage re-exploration at higher difficulty

**Mistakes made along the way:**
- threshold=0.8 was too easy — agent passed stages by luck without learning
- Initial epsilon=1.0 meant the network never participated in decisions early on; lowered to 0.3
- Stages 1–2–3 were too easy (passed in ~100 episodes each) but taught nothing; threshold raised to 0.95 to force real learning

---

## Final Results

| Stage | Empty Cells | Success Rate | Result |
|-------|-------------|--------------|--------|
| 1     | 3           | ~100%        | ✅ Passed |
| 2     | 6           | ~100%        | ✅ Passed |
| 3     | 10          | 96%          | ✅ Passed (took a long time) |
| 4     | 12          | 76–82%       | ❌ Stuck, declining |

---

## Bottlenecks

### 1. Catastrophic Forgetting
The success rate declined from 82% → 78% → 76% over time instead of improving.

**Cause:** Once epsilon decayed to 0.01, the agent stopped exploring. The replay buffer filled up with recent experience only, pushing out older good experiences. The network slowly unlearned what it had previously learned.

**Attempted fixes:**
- Raised `epsilon_min` from 0.01 to 0.05 → marginal improvement
- Expanded replay buffer from 10,000 to 50,000 → made things worse; early bad experiences diluted the good ones

**Conclusion:** Catastrophic forgetting is a fundamental issue with the Q-learning family. Hyperparameter tuning cannot fix it at the root.

### 2. DQN's Greedy Local Decision-Making
DQN learns "which action has the highest Q-value in the current state." But Sudoku requires global planning — what you fill in one cell affects every row, column, and box it belongs to. DQN's step-by-step greedy decisions are fundamentally misaligned with this kind of global constraint satisfaction.

### 3. Large Search Space
A 9×9 Sudoku has approximately 6.7×10²¹ possible fill combinations. Even with curriculum learning and dense rewards, DQN's exploration efficiency breaks down at higher difficulty levels.

---

## Why We Abandoned DQN

A literature review confirmed that no pure DQN has been shown to reliably solve complete 9×9 Sudoku puzzles. Successful approaches all use hybrid methods:
- **Neuro-Symbolic methods** (Neural Logic Machines): explicitly encode row/column/box constraints
- **Supervised learning**: train directly on known solutions
- **Imitation learning + RL fine-tuning**: build a foundation first, then optimize

The DQN ceiling in this experiment was approximately **76–82%**, with a declining trend due to catastrophic forgetting.

*Note: A separate imitation learning model was already trained and achieved 97% accuracy on 9×9 Sudoku — confirming the task is learnable, just not well-suited for pure DQN.*

---

## Next Step 🚀: Multi-Agent System + PPO