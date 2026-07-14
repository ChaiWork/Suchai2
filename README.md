# Pokémon TCG AI Battle Challenge Strategist

This repository contains reinforcement learning and rule-based agents developed for the Kaggle Pokémon TCG AI Battle Challenge using the `cabt` simulation engine.

## Codebase Structure

- `src/`: Main source folder containing RL agent components:
  - `agent.py`: MCTS planning and NN inference entry-points.
  - `model.py`: Model architecture and observation encoders.
  - `strategies/`: Tactical strategy guides for specific decks.
    - [mewtwo_strategy.md](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/src/strategies/mewtwo_strategy.md): Tactical guides, combo flows, and card lists for Team Rocket's Mewtwo ex.
- `Rulebasedmodel/`: Hardcoded rule-based agents for specific deck profiles:
  - `iono_agent.py`: Electric deck prioritizing voltaic chains.
  - `lucario_agent.py`: Fighting deck focusing on Mega Lucario ex.
  - `dragapult_agent.py`: Psychic/Fire deck focusing on Dragapult ex.
  - [mewtwo_agent.py](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/Rulebasedmodel/mewtwo_agent.py): Team Rocket's Mewtwo ex deck rule-based agent.
  - `main.py`: Entry-point script used for submissions. By default, copying one of the agents (e.g. `mewtwo_agent.py`) over `Rulebasedmodel/main.py` allows it to be evaluated.
- `decks/`: Pre-configured 60-card CSV files for various matchups.
- `battle_viewer/`: Local UI tools to parse battle logs and visualize matchups.

## Running Rule-Based Agents

To evaluate a specific rule-based agent:
1. Copy the desired agent code (e.g., `Rulebasedmodel/mewtwo_agent.py`) over the submission entry point `Rulebasedmodel/main.py` or the root `main.py`.
2. Ensure the matching `deck.csv` is placed at the root level of the workspace.
