# Pokémon TCG AI Battle Challenge — Documentation Index

Welcome to the technical documentation repository for the **Pokémon TCG AI Battle Challenge** autonomous agent system (**Team Rocket's Vanguard**).

---

## 1. Documentation Index

- [Kaggle Strategy Writeup (Publication Grade)](kaggle_strategy_writeup.md): The official, submission-ready strategy paper (1,844 words) strictly conforming to Kaggle's < 2,000 words limit and mapped directly to the competition evaluation rubric (70% Model, 20% Deck, 10% Report).
- [Media Gallery Manifest](media_gallery_manifest.md): Catalog of high-resolution visual exhibits (Figures 1–5), including dimensions, file sizes, recommended captions, analytical takeaways, and Pokémon IP fair use statements.

---

## 2. Core Operational Commands & Workflows

### 2.1 Agent Training & Reinforcement Learning
Runs self-play MCTS simulation rollouts, collects state-action transitions, and optimizes the Transformer Policy-Value Network (`MyModel`).

```powershell
# Run self-play training with active deck configuration (MEWTWO ex + Spidops)
python train.py --deck-config MEWTWO --epochs 30 --batch-size 128 --lr 1e-4

# Options:
#   --deck-config    Deck configuration profile [MEWTWO, DRAGAPULT, GRIMMSNARL]
#   --epochs         Number of training iterations (default: 26)
#   --batch-size     Minibatch size for policy-value gradient updates (default: 128)
#   --lr             AdamW learning rate (default: 1e-4)
#   --expert-weight  Initial weight for expert heuristic PUCT prior (default: 0.40)
```

### 2.2 Empirical Retreat & Energy Diagnostic Audits
Performs fine-grained replay inspection to verify the 9-point retreat audit and detect energy waste or suboptimal tactical retreats.

```powershell
# Run the 9-point retreat audit breakdown across recorded tournament replays
python audit_breakdown.py

# Run comprehensive deep replay analysis across all matches
python audit_deep.py --matches retreat_audit_matches.csv --threshold 0.95
```

### 2.3 Deck Visualizer & Composite Generation
Extracts card imagery and generates high-resolution composite deck visualizer images with card counts and category borders.

```powershell
# Run deck composite visualizer GUI
python extract_deck_visualizer.py

# Run in headless CLI mode for automated builds
python extract_deck_visualizer.py --cli --deck deck.csv --output deck_composite.png
```

### 2.4 Agent Testing & Rule-Based Benchmarking
Evaluates agent performance against local rule-based models and top competitor replicas.

```powershell
# Run agent test suite
python test_agent.py

# Test search strategy tutor routing
python test_search_strategy.py

# Benchmark agent against specific opponent bot
python test_mewtwo_profile.py --opponent Rulebasedmodel_Dragapult --games 20
```

### 2.5 Submission Packaging
Packages the agent code, model weights, and runtime configurations into a competition-compliant archive under the 197.7 MiB limit.

```powershell
# Package submission for Kaggle environment
python package_submission.py

# Output: submission.tar.gz (< 197.7 MiB limit strictly verified)
```
