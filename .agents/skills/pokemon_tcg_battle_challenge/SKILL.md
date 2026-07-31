---
name: pokemon_tcg_battle_challenge
description: "Comprehensive skill for understanding, modifying, training, and deploying the AlphaZero + MCTS + PPO Pokémon TCG AI agent codebase."
---

# Pokémon TCG AI Battle Challenge Codebase Skill

This skill provides AI agents with full architectural understanding of the **Pokémon TCG AI Battle Challenge Strategy** codebase—an **AlphaZero + Monte Carlo Tree Search (MCTS) + PPO Reinforcement Learning** agent built for competitive simulation in the Kaggle Pokémon TCG AI Battle Challenge.

---

## 🎯 Codebase Purpose & Core Objective

The primary goal of this repository is to build an autonomous AI agent capable of high-level strategic decision-making in the Pokémon Trading Card Game under hidden information.

* **Primary Engine**: `cabt` (Competitive AI Battle Tool) Python simulator.
* **RL Framework**: Policy-Value Neural Network (`MyModel` in `src/model.py`), Proximal Policy Optimization (PPO), Generalized Advantage Estimation (GAE), and GPU-accelerated parallel self-play workers.
* **Planning Engine**: Information-State Monte Carlo Tree Search (MCTS) with domain-specific heuristic prior guidance.

---

## 🏗️ Multi-Deck Architecture Overview

The codebase uses a modular, decoupled architecture supporting multiple deck archetypes cleanly via dynamic routing:

```
src/
├── configs/
│   └── active_deck.py          # Central setting: ACTIVE_DECK = "MEWTWO" | "LILLIE"
├── deck_profiles/
│   ├── base_profile.py         # Abstract Base Deck Profile (deck CSV path, card IDs)
│   ├── mewtwo_profile.py       # Team Rocket Mewtwo ex + Spidops metadata
│   └── lillie_profile.py       # Lillie's Clefairy ex + Togekiss metadata
├── reward_system/
│   ├── base_reward.py          # Generic RL rewards (Terminal Win/Loss, Prize diff, Donk guard)
│   ├── mewtwo_reward.py        # Team Rocket / Mewtwo specific rewards (Factory, Spidops, Articuno)
│   ├── lillie_reward.py        # Lillie's Clefairy ex rewards (Candy->Togekiss, Patch, Clefairy count)
│   └── reward_router.py        # Dynamic reward calculation router based on ACTIVE_DECK
├── expert_system/
│   ├── base_expert.py          # Board state context extraction (_extract_context)
│   ├── mewtwo_expert.py        # Team Rocket Mewtwo ex action prior heuristics
│   ├── lillie_expert.py        # Lillie's Clefairy ex + Togekiss action prior heuristics
│   └── expert_router.py        # Dynamic expert prior router returning (bonus, trigger) tuple
├── expert_knowledge.py         # Lightweight facade delegating to expert_router
├── agent.py                    # Kaggle simulation agent entry point (deck-agnostic)
└── training/
    ├── rewards.py              # Lightweight facade delegating to reward_router
    ├── worker.py               # Parallel self-play trajectory worker
    ├── evaluator.py            # Win-rate evaluation against 25+ rule-based bot opponents
    ├── inference_server.py     # Threaded GPU model inference server
    └── replay_buffer.py        # Prioritized Replay Buffer for PPO
```

---

## 🃏 Supported Deck Archetypes & Strategies

### 1. Lillie's Clefairy ex + Togekiss (`ACTIVE_DECK = "LILLIE"`)
* **Default Active Deck**: Set in `src/configs/active_deck.py` and defined in root [deck.csv](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/deck.csv).
* **Key Mechanics & Cards**:
  * **Lillie's Clefairy ex (`ID: 272`)**: Primary Basic attacker scaling with `{P}` Energy.
  * **Telepathic Psychic Energy (`ID: 19`)**: Attaching from hand to a `{P}` Pokémon immediately benches up to 2 Basic `{P}` Pokémon.
  * **Colress's Tenacity (`ID: 1194`)**: Supporter tutor searching 1 Stadium (**Mystery Garden `ID: 1263`**) + 1 Energy (**Telepathic Energy `ID: 19`**).
  * **Hilda (`ID: 1225`)**: Supporter tutor searching 1 Evolution Pokémon (**Togekiss `ID: 214`**) + 1 Energy.
  * **Rare Candy (`ID: 1079`)**: Evolves Togepi directly into Togekiss.
  * **Wondrous Patch (`ID: 1146`)**: Accelerates discarded `{P}` energy onto benched/active `{P}` Pokémon.
  * **Mystery Garden (`ID: 1263`)**: Discards 1 excess energy to refill hand up to `{P}` Pokémon in play count.
  * **Lillie's Pearl (`ID: 1172`)**: Tool equipment providing stat & prize protection.
  * **Rule Box Search Distinctions**:
    * **Ultra Ball (`ID: 1121`)**: Searches ANY Pokémon (including Rule-Box Clefairy ex).
    * **Poké Pad (`ID: 1152`)**: Searches non-Rule-Box Pokémon ONLY (Togepi, Togekiss, Smoochum, Psyduck).

### 2. Team Rocket Mewtwo ex + Spidops (`ACTIVE_DECK = "MEWTWO"`)
* **Key Mechanics & Cards**:
  * **Team Rocket Mewtwo ex (`ID: 431`)**: Primary high-HP attacker.
  * **Spidops (`ID: 401`)**: Trap & retreat lock utility.
  * **Articuno (`ID: 414`)**: Repelling Veil bench shield against Dragapult/Alakazam snipe.
  * **TR Factory (`ID: 1257`)**: Stadium draw engine for Team Rocket Pokémon.

---

## 🛠️ Key CLI Commands & Workflows

### 1. Run Self-Play Training (`train.py`)
```powershell
python train.py --epochs 15 --self-play-episodes 100 --batch-size 256 --lr 5e-5 --eval-episodes 120 --num-workers 4
```
* **Explicit Deck Switching**:
  ```powershell
  $env:ACTIVE_DECK="LILLIE"; python train.py
  $env:ACTIVE_DECK="MEWTWO"; python train.py
  ```

### 2. Evaluate Benchmark Win Rate (`test_agent.py`)
Runs 1-on-1 matches against rule-based bot opponents:
```powershell
python test_agent.py
```

### 3. Generate Composite Deck Visualizer (`extract_deck_visualizer.py`)
Generates high-resolution composite visualizer sheets from a deck list:
```powershell
python extract_deck_visualizer.py
```

### 4. Package Submission (`package_submission.py`)
Bundles `main.py`, `deck.csv`, weights, and helper scripts into `submission.tar.gz` (strictly under 197.7 MiB):
```powershell
python package_submission.py
```

---

## ⚠️ Important Rules & Best Practices

1. **Rule Box Constraints**: Poké Pad (`1152`) cannot target Rule-Box Pokémon like Lillie's Clefairy ex (`272`) or Mewtwo ex (`431`). Use Ultra Ball (`1121`), Telepathic Energy (`19`), or Supporters instead.
2. **Kaggle Environment Resolution**: All imports and file loaders must support dynamic path resolution for `/kaggle_simulations/agent/`.
3. **No Hardcoded Deck Assumptions**: `agent.py` must remain deck-agnostic. All deck-specific rewards and action priors MUST be placed inside `src/reward_system/` and `src/expert_system/`.
4. **Git Data Exclusions**: Never track `.pdf` or raw `.csv` dataset dumps exceeding Git limits.
