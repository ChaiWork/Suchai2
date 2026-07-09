---
name: pokemon_tcg_agent_developer
description: "Triggered when building, training, debugging, or deploying an AI agent for the Kaggle Pokémon TCG AI Battle Challenge using the cabt engine."
---

# Pokémon TCG AI Battle Challenge Agent Developer Skill

This skill guides the agent in building, training, and packaging a high-performance Pokémon TCG AI agent for the Kaggle Battle Challenge.

## Key Rules & Guidelines

1. **Submission Format Constraint**:
   * The submission must be a `.tar.gz` bundle with `main.py` and `deck.csv` at the root directory level (not nested).
   * Bundling command: `tar -czvf submission.tar.gz main.py deck.csv [any helper script/model file]`

2. **Kaggle Environment Directories**:
   * When executed on the competition server, your agent files are placed in `/kaggle_simulations/agent/`.
   * **Rule**: All custom python imports and file-loading commands (e.g. loading model weights or CSVs) must resolve correctly under this path. Use relative imports or check `/kaggle_simulations/agent/` if using absolute paths.

3. **Size Limits & Exclusions**:
   * The bundle must not exceed **197.7 MiB**.
   * Keep model weight sizes compact. 
   * Do NOT include the large game PDFs (`Card_ID List_EN.pdf`, `Card_ID List_JP.pdf`) in the submission. These are ignored via `.gitignore` and should never be packaged.

4. **Strategy & Model Design**:
   * A pure rule-based approach is rarely enough to win. Focus on:
     * **MCTS (Monte Carlo Tree Search)** for forward-planning card choices, coin tosses, and opponent moves.
     * **Reinforcement Learning** (e.g., Deep Q-Networks, Policy Gradients) to train policy and value estimators.
     * **Information State modeling** since opponent hands/decks are hidden.
   * Review [reinforcement-learning-and-mcts-sample-code.ipynb](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/reinforcement-learning-and-mcts-sample-code.ipynb) for a starting template.

5. **Reference Documentation**:
   * Refer to the full competition rules and details in [competition_info.md](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/competition_info.md).
   * Refer to the cabt simulator API documentation online at: https://matsuoinstitute.github.io/cabt/
