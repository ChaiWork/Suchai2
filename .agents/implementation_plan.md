# Implementation Plan: Convert Notebook to Modular Python Code

Convert the sample reinforcement learning and MCTS notebook code into a clean, modular Python codebase, structured for local training/evaluation and prepared for Kaggle Battle Challenge submission.

## User Review Required

> [!IMPORTANT]
> The original notebook combined model definition, MCTS agent search, self-play data collection, evaluation, and training into a single monolithic script. 
> To align with development best practices, we propose splitting the code into modular files:
> 1. `model.py`: Model architecture and state/action encoding (`SparseVector`).
> 2. `agent.py`: MCTS planning, random agent, and the competition entry-point `agent()` function.
> 3. `train.py`: Self-play data collection, evaluation, and PyTorch training loop.
> 4. `deck.csv`: The list of 60 card IDs used by the agent, satisfying TCG rules.
> 5. `main.py`: Entry-point file required by Kaggle that loads model weights, reads `deck.csv`, and runs `agent.py`.
> 6. `package_submission.py`: Utility script to build `submission.tar.gz`.

## Open Questions

None at this stage. The structure is based on standard Python development practices and the competition's submission constraints.

## Proposed Changes

We will create the new Python files at the workspace root directory level.

---

### Agent and Model Modules

#### [NEW] [model.py](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/model.py)
Defines the `DecoderLayer`, `MyModel`, the input vectorizer class `SparseVector`, and the helper functions `get_encoder_input` and `get_decoder_input` to convert simulator observations into PyTorch tensors.

#### [NEW] [agent.py](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/agent.py)
Implements the MCTS search algorithm (`Node`, `Child`, `create_node`, `mcts_agent`), the baseline `random_agent`, and the main competition entry point `agent(obs_dict: dict) -> list[int]`. It handles:
- Dynamically finding paths to model weights and deck files.
- Re-initializing/loading model weights only when needed.

#### [NEW] [deck.csv](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/deck.csv)
A CSV containing the 60 Card IDs representing the agent's deck (based on `sample_deck`).

---

### Training and Execution Entry Points

#### [NEW] [train.py](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/train.py)
The self-play and training loop. It will import modules from `model.py` and `agent.py` to:
- Run self-play simulation loops.
- Evaluate progress against a random agent.
- Run optimizer steps to update model weights.
- Save checkpoints to the `out/` directory.

#### [NEW] [main.py](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/main.py)
The official submission entry point. It sets up paths (handling Kaggle's `/kaggle_simulations/agent/` environment), loads `deck.csv` and the trained `model.pth`, and exposes the `agent(obs_dict: dict) -> list[int]` interface for the competition runner.

#### [NEW] [package_submission.py](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/package_submission.py)
A packaging script that bundles `main.py`, `deck.csv`, `model.pth`, `model.py`, `agent.py`, and the `cg` library into a single compliant `submission.tar.gz` under the 197.7 MiB limit.

## Verification Plan

### Automated Tests
- Run a single-step check using a mock observation or dry-run a local game using `cg.game` via a simple execution check:
  `python -c "import main; print('Main agent loaded successfully!')"`
- Run `train.py` for a single iteration (e.g. 1 evaluation battle, 1 self-play battle, and 1 batch of training) to ensure correctness of the entire training loop.

### Manual Verification
- Verify that `submission.tar.gz` can be generated and contains only the required source files and weights, keeping it well below the 197.7 MiB size limit.
