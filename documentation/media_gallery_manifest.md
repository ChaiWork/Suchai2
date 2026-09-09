# Media Gallery Manifest & Visual Documentation Guide

**Competition**: Kaggle Pokémon TCG AI Battle Challenge — Strategy Category  
**Agent**: Team Rocket's Vanguard (Hybrid AlphaZero + Transformer MCTS)  
**Target Submission Checkpoint**: `best_modelMEWTWO.pth` (Run 62 Suite)

---

## 1. Overview & Upload Guidelines

In the Kaggle Strategy Category submission portal, visual assets are uploaded through the **Media Gallery** panel rather than embedded as raw base64 strings in the writeup editor. The main writeup document (`documentation/kaggle_strategy_writeup.md`) explicitly cites each exhibit as **Figure 1** through **Figure 5**.

This manifest provides the exact metadata, image specifications, recommended captions, and strategic takeaways to copy-paste into the Kaggle Media Gallery modal during submission.

---

## 2. Visual Asset Catalog

### Figure 1: Team Rocket Mewtwo ex + Spidops / Battle Cage Deck Composition
- **File Name**: `deck_composite.png`
- **File Location**: `deck_composite.png` (Workspace root)
- **Image Specifications**: 1524 × 1114 px | 1,014.5 KB (PNG)
- **Citing Section in Writeup**: Section 2 (*Intentional Deck Construction*)
- **Recommended Kaggle Caption**:  
  > *Figure 1: Complete 60-card composition of the Team Rocket Mewtwo ex + Spidops / Battle Cage deck, illustrating primary multi-prize attackers, single-prize punishment lines, and dedicated tutor engines.*
- **Key Analytical Callouts for Judges**:
  - **Dual-Threat Synergy**: Showcases the deliberate pairing of TR Mewtwo ex (#431, heavy multi-prize burst) with TR Spidops (#401, single-prize punish trading scaling with TR cards in play).
  - **Spread Protection**: Highlights 3x Battle Cage (#1264), nullifying bench damage counters from top-tier threats like Dragapult ex *Phantom Dive*.
  - **Tutor Hierarchy**: Demonstrates 4x Transceiver, 4x Poké Pad, and 2x Ultra Ball, creating a zero-discard tutor pipeline that reserves Ultra Ball strictly for ex attackers.

---

### Figure 2: Executive Strategy Report & Comprehensive Evaluation Dashboard
- **File Name**: `strategy_report.png`
- **File Location**: `out/runs/run_62/strategy_report.png`
- **Image Specifications**: 4777 × 2752 px | 635.5 KB (PNG)
- **Citing Section in Writeup**: Section 3 (*Agent Architecture*) & Section 5 (*Empirical Validation*)
- **Recommended Kaggle Caption**:  
  > *Figure 2: Executive strategy report and evaluation dashboard displaying macro win-rate trajectories, opponent matchup heatmaps, and situational action conversion rates across tournament self-play.*
- **Key Analytical Callouts for Judges**:
  - **Macro Win-Rate Ceiling**: Visualizes the agent reaching and stabilizing at a ~70% win-rate against the broader tournament population.
  - **Matchup Generalization**: Highlights performance across 53 distinct opponent models with zero blind spots against tier-1 meta decks.
  - **Conversion Efficiency**: Demonstrates high rates of converting board advantage into decisive prize leads.

---

### Figure 3: Learning Dynamics & Convergence Metrics across Training Epochs
- **File Name**: `learning_curves.png`
- **File Location**: `out/runs/run_62/learning_curves.png`
- **Image Specifications**: 3977 × 1232 px | 235.9 KB (PNG)
- **Citing Section in Writeup**: Section 3 (*MCTS Search*) & Section 5 (*Training Dynamics*)
- **Recommended Kaggle Caption**:  
  > *Figure 3: Training telemetry showing policy loss convergence, value loss minimization (settling at 0.0329), mean return stabilization, and steady 60.0%–70.5% self-play win rates through Epoch 26.*
- **Key Analytical Callouts for Judges**:
  - **Policy & Value Convergence**: Smooth reduction in value loss demonstrates that the Transformer network developed accurate long-term game state evaluations.
  - **Stable Exploration**: Policy entropy and reference KL metrics show the network avoided catastrophic forgetting or policy collapse.
  - **Game Length Normalization**: Average game duration stabilizes near 20.3 turns, reflecting efficient prize-taking rather than stalling.

---

### Figure 4: Fine-Grained Reward Telemetry and Multi-Objective Contribution Breakdown
- **File Name**: `reward_breakdown.png`
- **File Location**: `out/runs/run_62/reward_breakdown.png`
- **Image Specifications**: 2960 × 2681 px | 214.8 KB (PNG)
- **Citing Section in Writeup**: Section 4 (*Strategic Reasoning & Algorithmic Innovations*)
- **Recommended Kaggle Caption**:  
  > *Figure 4: Fine-grained reward telemetry decomposing multi-objective signals: prize delta shaping, knockout bonuses, lethal detection scaling (+0.0138), search tempo, and retreat efficiency.*
- **Key Analytical Callouts for Judges**:
  - **Horizon Trap Mitigation**: Visualizes positive retreat efficiency and search tempo signals, proving the agent does not execute short-term actions that damage long-term board equity.
  - **Lethal Detection Spike**: Shows intentional reward amplification when opponent ex Pokémon enter knockout range, driving aggressive two-prize conversions.
  - **Deckout and Stall Penalties**: Demonstrates strong negative feedback keeping the agent focused on active prize acquisition.

---

### Figure 5: Action Distribution and Decision Entropy across Self-Play Generations
- **File Name**: `action_distribution.png`
- **File Location**: `out/runs/run_62/action_distribution.png`
- **Image Specifications**: 3561 × 1379 px | 194.9 KB (PNG)
- **Citing Section in Writeup**: Section 5 (*Action Diversity and Entropy*)
- **Recommended Kaggle Caption**:  
  > *Figure 5: Action distribution proportions across training generations, verifying balanced execution of attack (838/epoch), item play (4,055/epoch), energy attachment (1,702/epoch), evolution, and abilities.*
- **Key Analytical Callouts for Judges**:
  - **Action Diversity Index (0.964)**: Verifies that the agent executes a healthy, varied tactical repertoire without degenerating into passive passing.
  - **Retreat Regularity**: Demonstrates that retreat actions (~805/epoch) are disciplined and purposeful, aligning with the 9-point retreat audit findings.
  - **Evolution & Ability Ratios**: Confirms steady Stage 1 Spidops evolutions (~566/epoch) and ability activations (~1,085/epoch).

---

## 3. Intellectual Property & Fair Use Statement

```
DISCLAIMER ON POKÉMON INTELLECTUAL PROPERTY:
Pokémon, Pokémon character names, card designs, and card text are registered 
trademarks of Nintendo, Creatures Inc., and GAME FREAK inc. 

All card imagery and deck compositions presented in this submission are 
reproduced under the Fair Use doctrine for non-commercial academic research, 
technical analysis, and algorithmic benchmarking within the Kaggle Pokémon TCG 
AI Battle Challenge.
```
