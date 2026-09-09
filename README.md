# Pokémon TCG AI — Team Rocket's Vanguard

> **An Information-State AlphaZero agent for the Kaggle Pokémon TCG AI Battle Challenge.**
> Transformer-guided MCTS + PPO Self-Play, targeting 70%+ win rate with the Team Rocket Mewtwo ex control deck.

---

## Overview

**Team Rocket's Vanguard** is an autonomous Pokémon Trading Card Game AI agent built for the [Kaggle Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge).

The agent couples an **Information-State Monte Carlo Tree Search (AlphaZero MCTS)** with a **Transformer Policy-Value Network**, trained via asynchronous self-play + PPO against a diverse pool of 50+ rule-based opponent decks.

**Key results:** 60–70.5% win rate across 26 training epochs against 53 distinct opponent models, including top Kaggle competitor replicas.

---

## Features

| Component | Description |
|---|---|
| **Policy-Value Network** | Transformer encoder-decoder (`MyModel`): 3 layers, 8 heads, 36 card features per token |
| **PPO Training** | Proximal Policy Optimization with Generalized Advantage Estimation (GAE) |
| **Information-State MCTS** | AlphaZero PUCT with 200 simulations per move, handling hidden information |
| **Expert Knowledge** | Domain-specific strategic prior bonuses (`expert_router.py`) injected into MCTS PUCT formula |
| **Strategic Reward Shaping** | Multi-component reward system: terminal outcomes + prize trades + resource management |
| **Self-Play + League Play** | Parallel async workers + league of historical checkpoints as opponents |
| **SPRT Evaluation** | Sequential Probability Ratio Test for statistically rigorous model comparisons |
| **Elo Evaluation** | Online Elo tracking against 50+ opponent agents |

---

## Architecture

```
Game State (cabt engine)
         │
         ▼
Observation Encoder
 [Board Tokens + Hand + Discard + Energy — 36 features/card]
         │
         ▼
Transformer Encoder (3 layers, 8 heads)
         │
    ┌────┴────┐
    ▼         ▼
Policy Head  Value Head
(action      (scalar
 logits)      V(s) ∈ [-1,1])
    │         │
    └────┬────┘
         ▼
PUCT = Q(s,a) + c·P(s,a)·√N(s)/(1+N(s,a)) + Expert Bonus
         │
         ▼
MCTS Tree Search (200 simulations)
         │
         ▼
Selected Action
         │
         ▼
PPO Update ← Trajectory + Strategic Reward
```

### Expert Knowledge Integration

The expert prior bonus is **not** a hard rule — it is a soft multiplier on action probabilities:

```
UCT = Q(s,a) + c·P(s,a) + w·expert_bonus(obs, action)
```

The expert bonus decays with epoch (NN dominates over time) and scales with:
1. **Epoch scale** — decays as training progresses
2. **Entropy scale** — amplifies when NN is uncertain
3. **Risk scale** — amplifies in high-risk game states (powered retreat, Dragapult threat, etc.)

---

## Installation

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA-capable GPU (recommended)

```bash
git clone https://github.com/<your-username>/pokemon-tcg-ai.git
cd pokemon-tcg-ai
pip install -r requirements.txt
```

The `cg/` directory contains the pre-compiled `cabt` game engine binaries (Windows `.dll`, Linux `.so`, macOS `.dylib`). No separate installation is needed.

---

## Usage

### Train

```bash
python train.py --epochs 15 --self-play-episodes 100 --batch-size 256 --lr 5e-5 \
                --eval-episodes 120 --num-workers 4
```

Optional environment variables:
```bash
# Redirect PyTorch cache to a different drive (useful on limited-space systems)
$env:POKEMON_AI_TORCH_HOME = "E:/torch_cache"
$env:POKEMON_AI_TMPDIR     = "E:/temp"
python train.py
```

### Evaluate

```bash
python test_agent.py
```

Runs 2 games against each of the 50+ rule-based opponent decks and reports:
- Win / loss counts per opponent
- Overall win rate (easy / hard / combined)
- Average turns per game

### Package for Kaggle Submission

```bash
python package_submission.py
```

Bundles `main.py`, `deck.csv`, model weights, `src/`, and `cg/` into `submission.tar.gz` (must be ≤ 197.7 MiB).

---

## Configuration

### Deck Selection

The active deck is configured in [`src/configs/active_deck.py`](src/configs/active_deck.py):

```python
ACTIVE_DECK = "MEWTWO"   # Team Rocket Mewtwo ex + Spidops / Battle Cage
```

### Training Hyperparameters (CLI)

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 5 | Number of training epochs |
| `--self-play-episodes` | 100 | Self-play games per epoch |
| `--self-play-ratio` | 0.5 | Fraction of self-play vs rule-based games |
| `--batch-size` | 128 | PPO mini-batch size |
| `--lr` | 5e-5 | AdamW learning rate |
| `--buffer-size` | 20000 | Prioritized replay buffer capacity |
| `--eval-episodes` | 50 | Evaluation games per epoch |
| `--patience` | 10 | Early stopping patience (win-rate) |
| `--num-workers` | auto | Parallel self-play workers |

### Expert Knowledge

Expert guidance is controlled in [`src/expert_system/base_expert.py`](src/expert_system/base_expert.py):

```python
USE_EXPERT_GUIDANCE = True
EXPERT_WEIGHT       = 1.5   # Scale applied to all expert bonus values
```

---

## Model Checkpoints

Trained weights are **not committed** to this repository due to size (~306 MB). See [`models/README.md`](models/README.md) for:
- How to obtain the pre-trained checkpoint
- How to load it for inference
- Where to place it for submission packaging

---

## Project Structure

```
pokemon-tcg-ai/
│
├── main.py                      # Kaggle simulation entry point
├── train.py                     # PPO self-play training loop
├── test_agent.py                # Benchmark evaluation vs rule-based bots
├── package_submission.py        # Kaggle submission packager
├── extract_deck_visualizer.py   # Deck card composite visualization utility
├── deck.csv                     # Active 60-card deck definition
├── deck_mewtwo.csv              # Mewtwo ex reference deck
├── visualizer.html              # Local battle replay viewer
│
├── src/
│   ├── agent.py                 # MCTS agent + GPUInferenceClient
│   ├── model.py                 # Transformer Policy-Value Network (MyModel)
│   ├── expert_knowledge.py      # Expert system facade
│   │
│   ├── configs/
│   │   └── active_deck.py       # Active deck configuration
│   │
│   ├── deck_profiles/
│   │   ├── base_profile.py      # Abstract deck profile interface
│   │   └── mewtwo_profile.py    # Mewtwo ex + Spidops card IDs and metadata
│   │
│   ├── expert_system/
│   │   ├── base_expert.py       # Expert scale / confidence parameters
│   │   ├── expert_router.py     # Routes to deck-specific expert heuristics
│   │   ├── mewtwo_expert.py     # Team Rocket Mewtwo ex action priors
│   │   ├── energy_evaluator.py  # Generic energy state evaluation
│   │   └── search_strategy.py   # Search/tutor card strategy helpers
│   │
│   ├── reward_system/
│   │   ├── base_reward.py       # Generic RL rewards (terminal, prize, donk)
│   │   ├── mewtwo_reward.py     # Mewtwo-specific strategic rewards
│   │   └── reward_router.py     # Unified reward entry point
│   │
│   ├── training/
│   │   ├── worker.py            # Parallel self-play trajectory worker
│   │   ├── evaluator.py         # Win-rate evaluation + SPRT + Elo
│   │   ├── checkpoint.py        # Atomic checkpoint manager
│   │   ├── inference_server.py  # GPU batch inference server
│   │   ├── replay_buffer.py     # Prioritized Replay Buffer (PPO)
│   │   ├── gae.py               # Generalized Advantage Estimation
│   │   ├── logger.py            # Metrics logger + progress bar
│   │   ├── rewards.py           # Reward engine facade
│   │   └── card_database.py     # Card data helpers
│   │
│   ├── strategies/
│   │   ├── mewtwo_strategy.md   # Team Rocket Mewtwo ex tactical guide
│   │   └── mewtwo_matchup_strategy.md  # Matchup-specific notes
│   │
│   └── plot_metrics.py          # Training metrics visualization
│
├── cg/                          # cabt game engine (pre-compiled binaries + Python API)
│   ├── api.py                   # Game observation/action API
│   ├── game.py                  # Battle start/finish/select wrappers
│   ├── cg.dll / libcg.so / libcg.dylib  # Platform-specific engine binaries
│   └── sim.py / utils.py
│
├── decks/                       # Rule-based opponent decks for evaluation
│   ├── easy/                    # Easy tier opponents
│   ├── hard/                    # Hard tier opponents (top-player replicas)
│   └── DRAGOPULT/               # Dragapult reference deck
│
├── models/
│   └── README.md                # Checkpoint download & loading instructions
│
├── documentation/
│   ├── intro.md                 # Documentation index
│   └── kaggle_strategy_writeup.md   # Full competition writeup
│
└── .agents/
    └── skills/                  # AI agent skills and rules
```

---

## Reproducibility

| Parameter | Value |
|---|---|
| Random seed | `42` (set in `train.py`) |
| Python version | 3.10+ |
| PyTorch version | 2.x (see `requirements.txt`) |
| CUDA | Required for GPU training; CPU fallback supported |
| OS | Windows 10/11 (primary), Linux (Kaggle environment) |
| Hardware | Training: NVIDIA GPU (≥8 GB VRAM recommended) |

The `train.py` script sets `random.seed(42)`, `torch.manual_seed(42)`, and `torch.cuda.manual_seed_all(42)` before training. Training is **not perfectly deterministic** due to CUDA non-determinism in multi-process self-play workers.

---

## Evaluation Methodology

The agent is evaluated via two complementary methods:

### 1. Win-Rate Benchmark (`test_agent.py`)
- 2 games per opponent, across 50+ rule-based bot decks
- Reports easy/hard/combined win rates and Wilson score confidence intervals

### 2. SPRT (Sequential Probability Ratio Test) — `src/training/evaluator.py`
- Statistically rigorous model comparison during training
- Null hypothesis: new model ≤ baseline win rate (H₀ threshold: 0.45)
- Alternative hypothesis: new model ≥ target win rate (H₁ threshold: 0.55)
- α = 0.05, β = 0.10

### 3. Elo Tracking
- Online Elo updates per evaluation game (K = 32)
- Starting Elo: 1000 per model version

---

## Limitations

- **Hidden information**: The agent uses information-state MCTS sampling across plausible opponent hands, which is an approximation — not perfect hidden-state planning.
- **Out-of-distribution cards**: Cards not seen during training may produce sub-optimal embeddings (zero-initialized tokens).
- **Single deck focus**: The agent is specialized for Team Rocket Mewtwo ex. Generalizing to other decks requires retraining with deck-specific reward and expert modules.
- **Stochasticity**: Coin flips and shuffle order introduce irreducible variance. Win rates represent statistical averages over many games.
- **Compute requirements**: Full training (~15 epochs, 100 self-play games/epoch) requires a CUDA GPU and several hours.

---

## License

See [LICENSE](LICENSE) for details.
