# Pokémon TCG AI

> A reinforcement learning agent for strategic decision-making in Pokémon Trading Card Game environments using Information-State MCTS, PPO, self-play, and expert-guided reward shaping.

---

## 1. Overview

This project implements an autonomous artificial intelligence agent engineered for the [Kaggle Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge). The agent is designed to play full 60-card Pokémon Trading Card Game matches using the Team Rocket Mewtwo ex control deck archetype.

### The Challenge of Pokémon TCG for AI
Pokémon TCG presents a complex game-theoretic environment characterized by:
* **Imperfect Information**: Hidden opponent hand cards, facedown prize cards, and randomized deck orders create a non-Markovian decision process with vast information sets.
* **Stochasticity & Branching**: Coin flips, card shuffling, and damage variability generate non-deterministic state transitions, leading to combinatorial action branching (e.g., search selections, target assignments, multi-card discards).
* **Severe Resource Asymmetry**: Energy attachments are strictly limited to one per turn under standard rules. Suboptimal resource allocation—such as premature retreats or poorly timed tutor searches—can permanently surrender tempo.
* **Delayed Reward Signals**: Matches often last 15 to 30 turns across dozens of individual micro-actions. Terminal win/loss outcomes alone provide insufficient gradient feedback for deep networks.

### The AI Approach
To address these challenges, our architecture combines:
1. **Information-State Monte Carlo Tree Search (MCTS)**: Employs belief-state determinization by sampling hidden cards to conduct tree rollouts governed by Upper Confidence Bounds for Trees (PUCT).
2. **Transformer Policy-Value Network (`MyModel`)**: A 26.65M parameter neural network featuring a 3-layer Transformer Encoder and a Transformer Decoder layer with static 36-dimensional dense card embeddings and dynamic board state tokenization.
3. **Reinforcement Learning via Self-Play & PPO**: Multi-worker asynchronous self-play against diverse rule-based agents and league historical checkpoints, optimized using Generalized Advantage Estimation (GAE) and Prioritized Experience Replay.
4. **Expert Domain Guidance & Strict Vetoes**: A modular heuristic coaching layer providing bounded prior bonuses and strict action vetoes for high-risk board states (e.g., preventing energy waste on active carry attackers).

---

## 2. Key Features

* **Information-State AlphaZero MCTS**: Determinizes hidden opponent cards from empirical belief states and guides exploration via PUCT with root Dirichlet noise.
* **Transformer Policy-Value Architecture**: 26,655,234 trainable parameters combining EmbeddingBag tokenization, a 3-layer Transformer Encoder, a 1-layer Transformer Decoder, and dual output heads.
* **Domain Feature Engineering**: 36-dimensional static feature vector per card encoding HP, energy types, stages, rule-box tags (ex, Mega, Tera, ACE SPEC), and specific energy-cost requirements.
* **PPO / AlphaZero Optimization**: Huber value loss, cross-entropy policy loss with label smoothing, adaptive policy entropy bonuses, and gradient norm clipping.
* **Generalized Advantage Estimation (GAE)**: Implemented with $\gamma = 0.995$ and $\lambda = 0.98$, complete with running standard deviation normalization for stable TD error prioritization.
* **Prioritized Replay Buffer**: Proportional TD-error sampling with annealed importance-sampling exponent $\beta \in [0.4, 1.0]$.
* **Asynchronous Multi-Worker Self-Play**: Multi-process worker pool serviced by a centralized batched GPU inference server (`GPUInferenceServer`).
* **Curriculum League Training**: Historical checkpoint pool with Gaussian Elo sampling against 53 distinct opponent models, including top Kaggle competitor replicas.
* **Sequential Probability Ratio Test (SPRT)**: Wald SPRT evaluation with log-likelihood ratio tracking ($\alpha = 0.05, \beta = 0.10$) and 95% Wilson score confidence intervals.
* **Dynamic Expert Guidance**: Adaptive heuristic coach scaling by epoch decay, policy entropy, and situational board risk.

---

## 3. System Architecture

```mermaid
flowchart TD
    subgraph Environment ["Pokémon TCG Game Engine (cg-lib)"]
        OBS[Raw Observation obs_dict]
        API[C++ Engine cg.api / cg.game]
    end

    subgraph StateRepresentation ["State Tokenization"]
        OBS --> ENC_INP[Encoder SparseVector: Hand, Active, Bench, Discard]
        OBS --> DEC_INP[Decoder SparseVector: Available Legal Actions]
        FEAT[36 Dense Card Features] --> EMB[EmbeddingBag + Feature Projection]
        FEAT --> ENC_INP
        FEAT --> DEC_INP
    end

    subgraph NeuralNetwork ["Transformer Policy-Value Network (MyModel: 26.65M Params)"]
        ENC_INP --> EMB
        EMB --> TRANS_ENC[Transformer Encoder: 3 Layers, 4 Heads, d=256]
        TRANS_ENC --> VAL_HEAD[Value Head: Scalar V in [-1, +1]]
        TRANS_ENC --> TRANS_DEC[Transformer Decoder: 1 Layer, 4 Heads, d=256]
        DEC_INP --> TRANS_DEC
        TRANS_DEC --> POL_HEAD[Policy Head: Action Logits]
    end

    subgraph SearchDecision ["MCTS & Expert Guidance"]
        POL_HEAD --> LOGITS[Policy Logits]
        EXP[Expert System: Mewtwo Expert] -->|Risk & Entropy Scaled Bonus| PRIOR[PUCT Action Priors]
        LOGITS --> PRIOR
        VAL_HEAD --> MCTS[Information-State MCTS: Determinization + Rollouts]
        PRIOR --> MCTS
        MCTS --> ACTION[Selected Action]
    end

    subgraph TrainingLoop ["Self-Play & Reinforcement Learning"]
        ACTION --> API
        API --> NEXT_STATE[State Transition S to S']
        NEXT_STATE --> REWARD[Reward Engine: Base + Mewtwo Specific]
        REWARD --> GAE_BUF[GAE Return & Advantage Calculation]
        GAE_BUF --> REPLAY[Prioritized Replay Buffer]
        REPLAY --> PPO_UPDATE[PPO / AlphaZero Optimization Step]
        PPO_UPDATE -->|Update Weights| NeuralNetwork
    end
```

---

## 4. AI Approach

### PPO / AlphaZero Policy Optimization
The agent learns policy distributions and state values through iterative policy evaluation and policy improvement:
* **Policy Loss**: Cross-entropy between the model's predicted action log-probabilities and the MCTS visit count distribution target, regularized with 5% label smoothing:
  $$\mathcal{L}_{\text{policy}} = -\sum_{a} \left[ (1 - \epsilon) \pi_{\text{MCTS}}(a) + \frac{\epsilon}{K} \right] \log \pi_\theta(a)$$
* **Value Loss**: Huber loss with $\delta = 1.0$ comparing predicted scalar outcomes $V_\theta(s)$ against bounded returns $z \in [-1, 1]$ generated by GAE:
  $$\mathcal{L}_{\text{value}} = \text{HuberLoss}(V_\theta(s), z; \delta=1.0)$$
* **Total Combined Loss**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + 0.25 \cdot \mathcal{L}_{\text{value}} - c_{\text{entropy}} \cdot \mathcal{H}(\pi_\theta)$$
  Where $c_{\text{entropy}}$ decays over training epochs ($0.035 \to 0.025 \to 0.015$), and reference model KL divergence is monitored to detect drift.
* **GAE Advantage Estimation**: Computes generalized advantages using discount factor $\gamma = 0.995$ and trace decay $\lambda = 0.98$:
  $$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t), \quad \hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$
  Advantages are standardized across the trajectory and assigned as sample priority in the replay buffer.

### Information-State Monte Carlo Tree Search (MCTS)
Because opponent hands, deck configurations, and prize cards are hidden, standard deterministic tree search cannot be directly applied.
* **Belief State Determinization**: Before tree expansion, `mcts_agent()` samples unrevealed cards from the known remaining deck pool to construct an explicit `SearchState` via `cg.api.search_begin()`.
* **PUCT Tree Traversal**: Nodes are selected using the polynomial Upper Confidence Bound:
  $$U(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
  Where $c_{\text{puct}} = 0.4 \cdot \sqrt{N(s)}$ and $P(s, a)$ is the prior probability from the network combined with expert guidance.
* **Dirichlet Exploration Noise**: During training self-play, Dirichlet noise ($\alpha = 0.30, \epsilon = 0.25$) is injected into the root node action distribution to ensure exploration of tactical variations.
* **Inference Efficiency**: In training self-play, the agent executes up to 200 rollouts per turn. In Kaggle competition deployment (`agent.py`), `search_count = 0` enables sub-5ms direct neural prior evaluation to eliminate timeout risk.

### Neural Network Architecture (`MyModel`)
The neural network represents board configurations through learned token embeddings combined with static domain features:

| Hyperparameter | Value | Description |
|---|---|---|
| **Model Dimension ($d_{\text{model}}$)** | 256 | Embedding and hidden layer dimension |
| **Attention Heads ($n_{\text{heads}}$)** | 4 | Multi-head self-attention heads |
| **Feedforward Dimension ($d_{\text{ff}}$)** | 512 | Inner dimension of Transformer feedforward layers |
| **Encoder Layers** | 3 | Standard PyTorch `TransformerEncoderLayer` stack |
| **Decoder Layers** | 1 | Custom `DecoderLayer` with multi-head cross-attention |
| **Total Parameters** | 26,655,234 | Verified parameter count (all trainable) |
| **Encoder Vocabulary Size** | 22,000 | Sparse token space for board entities and states |
| **Card Feature Dimensions** | 36 | Dense physical properties per card |

#### Input Representation & Dense Card Embeddings
Each card in the game database is pre-processed into a 36-dimensional feature vector projected to $d_{\text{model}}$:
* **Card Type**: One-hot vector across 7 types (Pokémon, Trainer, Energy, etc.).
* **Energy Type**: One-hot vector across 11 energy affinities (Grass, Psychic, Colorless, etc.).
* **Health Points (HP)**: Normalized value ($HP / 400.0$).
* **Evolutionary Stage**: Flags for Basic, Stage 1, and Stage 2.
* **Special Subtypes**: Flags for ex, Mega ex, Tera, and ACE SPEC cards.
* **Retreat Cost**: Normalized value ($\text{cost} / 4.0$).
* **Attack Metrics**: Attack count, maximum damage normalized by 300, and skill flags.
* **Energy Cost Requirements**: Exact minimum attack costs partitioned into Grass, Psychic, and Colorless requirements.

The observation state is mapped to indices and values via `SparseVector` objects and ingested by PyTorch `EmbeddingBag` layers with summation mode.

---

## 5. Expert Knowledge System

The expert system acts as a **domain coach** rather than a rule-based override. It computes an additive logit bias on MCTS action prior probabilities before the softmax transformation:

$$\text{logit}(a) \leftarrow \text{logit}(a) + 8.0 \cdot \text{bonus}_{\text{expert}}(s, a) \cdot w_{\text{final}}$$

### Confidence Scaling Formula
Expert influence dynamically adapts to the state of training and game context:

$$\text{Confidence} = \text{Scale}_{\text{epoch}} \times \text{Scale}_{\text{uncertainty}} \times \text{Scale}_{\text{risk}}$$

1. **Epoch Decay ($\text{Scale}_{\text{epoch}}$)**:
   * Epoch 1–20: $1.00$ (Full coaching during curriculum initialization)
   * Epoch 21–50: $0.70$ (Balanced collaboration between expert and network)
   * Epoch 51–100: $0.40$ (Neural network dominance; expert acts as safety net)
   * Epoch 101–200: $0.25$ (High-risk guidance only)
   * Epoch 200+: $0.15$ (Minimum retention floor)
2. **Uncertainty Scaling ($\text{Scale}_{\text{uncertainty}}$)**:
   $$\text{Scale}_{\text{uncertainty}} = 0.70 + 0.60 \times \mathcal{H}_{\text{normalized}}(\pi_{\text{NN}})$$
   When the neural network exhibits high policy entropy (uncertainty), expert coaching is amplified up to $1.30\times$. When the network is confident, coaching is scaled down to $0.70\times$.
3. **Situational Risk Multiplier ($\text{Scale}_{\text{risk}} \in [1.0, 1.60]$)**:
   Amplifies coaching under critical board conditions:
   * **Active Carry Retreat Risk ($1.60\times$)**: Active Mewtwo ex or Spidops possessing $\ge 2$ energy attempting to retreat.
   * **Misallocated Energy Attachment ($1.40\times$)**: Attaching energy to defensive support (Articuno) while main attackers require power.
   * **Opponent Bench Snipe Hazard ($1.30\times$)**: Opponent field contains Dragapult ex / Dreepy lines while Battle Cage is absent.
   * **Deckout Risk ($1.25\times$)**: Remaining deck cards $\le 5$.
   * **Incomplete Bench Setup ($1.20\times$)**: Fewer than 3 Team Rocket Pokémon in play.

### Strict Negative Vetoes
While positive bonuses are subject to epoch and entropy decay, **negative vetoes** ($\text{bonus} < 0$) are applied with fixed intensity ($\text{EXPERT\_WEIGHT} = 0.25$) to permanently disincentivize catastrophic blunders.

---

## 6. Reward Engineering

The reinforcement learning objective is shaped through dense, orthogonal transition signals calculated over consecutive states $S \to S'$.

```
Total Step Reward = Clip( Σ Base Strategic Components + Mewtwo Specific Reward, min=-0.50, max=+0.50 )
```

### 1. Base Strategic Signals (`src/reward_system/base_reward.py`)
* **Knockout Execution (`r_knockout`)**: $+0.40 \times \Delta \text{PrizesTaken}$.
* **Damage Efficiency (`r_damage_eff`)**: Up to $+0.20$ proportional to opponent HP reduced ($\Delta HP / 200.0$).
* **Lethal Execution (`r_lethal_detection`)**: $+0.35$ bonus when reducing active opponent HP to 0.
* **Attack Readiness (`r_attack_ready`)**: $+0.20$ when active reaches minimum required energy; $-0.25$ penalty for over-attaching to an already powered attacker.
* **Bench Backup Preparation (`r_backup_ready`)**: $+0.35$ whenever energy attached to benched attackers increases.
* **Bench Setup & Expansion (`r_bench_setup`)**: $+0.25$ upon achieving a full 5-Pokémon bench to maximize Spidops *Rocket Rush* damage scaling.
* **Evolution Progression (`r_evolution_progress`)**: $+0.30$ for evolving Tarountula into Spidops.
* **Stadium Board Value (`r_stadium_value`)**: $+0.20$ for deploying Battle Cage against spread-damage decks.
* **Retreat Efficiency (`r_retreat_eff`)**: $+0.25$ for pivoting a damaged attacker to safety; $-0.35$ for discarding scarce energy on healthy attackers.

### 2. Mewtwo Deck-Specific Safety Guards (`src/reward_system/mewtwo_reward.py`)
* **Turn 1 Supporter Penalty**: $-0.08$ for attempting illegal or wasted turn-1 supporters when going first.
* **Exposed Mewtwo ex Penalty**: $-0.08$ for starting Mewtwo ex active on turn 1 without setup.
* **Under-Power Attack Penalty**: $-0.08$ for attacking with Mewtwo ex without sufficient energy.
* **Spidops Unnecessary Retreat**: $-0.10$ for retreating an active Spidops that is ready to attack.
* **Four Rocket Pokémon Milestone**: $+0.04$ reward upon achieving 4 Team Rocket Pokémon on board.

---

## 7. Training Pipeline

```
┌────────────────────────────────────────────────────────┐
│                   Initialize Model                     │
│       MyModel (26.65M Params, AdamW lr=5e-5)           │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│         Multi-Worker Asynchronous Self-Play            │
│  - N Workers running cg-lib game engine                │
│  - Batched GPU Inference Server servicing queries      │
│  - Opponents: Current Self, League Pool, 53 Bot Decks  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│           Trajectory & Advantage Processing            │
│  - Step rewards: Base Strategic + Mewtwo Specific      │
│  - Generalized Advantage Estimation (GAE γ=0.995, λ=0.98)│
│  - TD-error priority assignment                        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Prioritized Replay Buffer                │
│  - Capacity: 20,000 samples                            │
│  - Proportional sampling with annealed β in [0.4, 1.0] │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│           PPO / AlphaZero Optimization Step            │
│  - Huber value loss (weight: 0.25)                     │
│  - Smoothed CE policy loss over MCTS visit targets     │
│  - Policy entropy bonus (decaying 0.035 -> 0.015)      │
│  - Mixed precision (torch.amp) + Grad clip (max_norm=1)│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               Evaluation & League Update               │
│  - Sequential Probability Ratio Test (SPRT) vs Tier 1  │
│  - Dynamic Elo updates (K=32)                          │
│  - Atomic checkpointing: best_model.pth, latest_model  │
└────────────────────────────────────────────────────────┘
```

---

## 8. Evaluation Methodology

The agent's strategic capability is assessed across three statistical layers:

1. **Sequential Probability Ratio Test (SPRT)**:
   * Compares the active training checkpoint against baseline opponents.
   * Null Hypothesis $H_0: p \le 0.50$ vs Alternative Hypothesis $H_1: p \ge 0.54$.
   * Error bounds: Type I error $\alpha = 0.05$, Type II error $\beta = 0.10$.
   * Log-likelihood ratio boundaries:
     $$A = \ln\left(\frac{\beta}{1 - \alpha}\right), \quad B = \ln\left(\frac{1 - \beta}{\alpha}\right)$$
   * Models reaching $LLR \ge B$ are accepted early; models falling below $A$ trigger early stopping counters.
2. **Wilson Score Confidence Intervals**:
   * Reports binomial win rates with 95% confidence bounds ($z = 1.96$) to account for small evaluation sample sizes.
3. **League Elo Rating System**:
   * Evaluates checkpoints in an ongoing historical league pool.
   * Checkpoint Elo ratings update with a standard $K=32$ factor based on self-play outcomes.

---

## 9. Evaluation Results

The metrics below represent verified empirical results recorded in [`out/runs/run_62/training_metrics.csv`](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/out/runs/run_62/training_metrics.csv) and [`out/league/elo.csv`](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/out/league/elo.csv) over 26 training epochs:

### Training Progress & Evaluation Win Rates

| Epoch | Win Rate vs Opponents | Average Game Length (Turns) | Value Loss (Huber) | Policy Loss (CE) | Explained Variance | Active League Elo |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **20** | 60.5% | 20.15 | 0.0302 | 0.1868 | 0.389 | 23,257.3 |
| **21** | 64.5% | 20.45 | 0.0368 | 0.2344 | 0.344 | 23,248.5 |
| **22** | **69.5%** | 20.15 | 0.0268 | 0.1853 | 0.293 | 23,288.8 |
| **23** | 58.5% | 21.31 | 0.0318 | 0.2043 | 0.240 | 23,322.9 |
| **24** | 64.0% | 20.76 | 0.0311 | 0.2065 | 0.250 | 23,392.5 |
| **25** | 61.5% | 19.65 | 0.0375 | 0.2240 | 0.177 | 23,483.1 |
| **26** | **69.5%** | 20.36 | 0.0330 | 0.2371 | 0.234 | 23,483.1 |

*Peak evaluation win rate across training runs reached **70.5%** against diverse opponent decks.*

### Empirical Matchup Performance (`out/runs/run_62/deck_matchup.csv`)

| Opponent Deck Archetype | Classification | Verified Win Rate | Strategic Context |
|---|---|:---:|---|
| `Rulebasedmodel_Dragapult` | Hard Meta Replica | **100.0%** (2/2) | Neutralized by Battle Cage bench immunity |
| `Rulebasedmodel_Hydrapple_ex` | Hard Meta Replica | **100.0%** (2/2) | Controlled via Mimikyu ex Safeguard |
| `Rulebasedmodel_Garchomp_ex` | Hard Meta Replica | **100.0%** (2/2) | Outpaced by Spidops 1-prize trading |
| `Rulebasedmodel_Alakazam` | Hard Meta Replica | **100.0%** (2/2) | High-burst Mewtwo ex carry resolution |
| `Rulebasedmodel_Lopunny` | Tier 1 Curriculum | **100.0%** (2/2) | Early bench establishment |
| `Rulebasedmodel_Crustle` | Tier 1 Curriculum | **100.0%** (2/2) | Damage efficiency and spacing |
| `Rulebasedmodel_Honchkrow` | Tier 1 Curriculum | **50.0%** (1/2) | Balanced trade match |
| `Rulebasedmodel_Abomasnow` | Tier 1 Curriculum | **50.0%** (1/2) | Weather attrition contest |
| `Rulebasedmodel_Grimmsnarl_ex`| Control Archetype | **0.0%** (0/2) | Hand disruption challenge (focus of active tuning) |

---

## 10. Pokémon TCG Deck Construction

The active deck is defined in [`deck.csv`](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/deck.csv) and [`deck_mewtwo.csv`](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/deck_mewtwo.csv):

```
Team Rocket Mewtwo ex + Spidops / Battle Cage (60 Cards Total)
```

| Card ID | Card Name | Count | Role in Deck Strategy |
|:---:|---|:---:|---|
| **431** | Team Rocket's Mewtwo ex | 2 | **Primary Carry**: Delivers massive *Erasure Ball* burst damage to close matches. |
| **401** | Team Rocket's Spidops | 4 | **Secondary Attacker**: Stage 1 attacker scaling single-prize damage via *Rocket Rush*. |
| **400** | Team Rocket's Tarountula | 4 | **Basic Stage 0**: Evolution foundation for Spidops. |
| **434** | Team Rocket's Mimikyu | 2 | **Defensive Shield**: *Safeguard* ability blocks damage from opposing Pokémon ex. |
| **414** | Team Rocket's Articuno | 2 | **Utility Pivot**: High-HP bench anchor and tactical damage absorber. |
| **1264** | Battle Cage | 3 | **Key Stadium**: Prevents damage counters on bench, neutralizing Dragapult *Phantom Dive*. |
| **1134** | Team Rocket's Transceiver | 4 | **Item Engine**: Universal tutor fetching any Team Rocket Supporter card. |
| **1152** | Poké Pad | 4 | **Zero-Cost Search**: Searches non-ex Basic Pokémon without hand discard cost. |
| **1094** | Bug Catching Set | 3 | **Grass Search**: Digs for Grass Pokémon and Basic Grass Energy. |
| **1121** | Nest Ball | 2 | **Bench Setup**: Direct bench tutor for Basic Pokémon. |
| **1086** | Buddy-Buddy Poffin | 1 | **Early Engine**: Tutors low-HP Basics onto the bench. |
| **1097** | Night Stretcher | 2 | **Resource Recovery**: Recycles key Pokémon or energy from discard to hand. |
| **1159** | Hero's Cape | 1 | **ACE SPEC Tool**: Provides +100 maximum HP to active carry. |
| **1175** | Brave Bangle | 2 | **Offensive Tool**: Increases attack damage against opposing Pokémon ex. |
| **1216** | Team Rocket's Ariana | 4 | **Supporter Draw**: Primary draw accelerator and hand refresh. |
| **1218** | Team Rocket's Giovanni | 2 | **Supporter Disruption**: Hand and board disruption. |
| **1220** | Team Rocket's Proton | 2 | **Supporter Acceleration**: Energy and item retrieval. |
| **1219** | Team Rocket's Petrel | 1 | **Supporter Recovery**: Recovers Team Rocket cards from discard. |
| **1227** | Lillie's Determination | 3 | **Supporter Draw**: Consistent hand replenisher. |
| **15** | Team Rocket's Energy | 4 | **Special Energy**: Provides double energy value for Team Rocket Pokémon. |
| **1** | Basic {G} Energy | 8 | **Basic Energy**: Grass energy fueling Spidops and attack costs. |

---

## 11. Project Structure

```text
pokemon-tcg-ai-battle-challenge-strategy/
├── cg/                             # Pre-compiled C++ game engine binaries (cabt)
│   ├── api.py                      # Python API wrapper for engine observation and actions
│   ├── game.py                     # Game lifecycle (battle_start, battle_select, battle_finish)
│   ├── libcg.so                    # Linux dynamic library
│   ├── libcg.dylib                 # macOS dynamic library
│   └── cg.dll                      # Windows dynamic library
├── models/                         # Checkpoint documentation and release guides
│   └── README.md                   # Instructions for downloading pre-trained weights
├── out/                            # Training artifacts, logs, and league models
│   ├── league/                     # League historical checkpoints and elo.csv
│   └── runs/                       # Individual training run logs, metrics, and plots
├── src/                            # Core agent and training source code
│   ├── configs/                    # Deck and configuration files
│   │   └── active_deck.py          # Active deck loader configuration
│   ├── deck_profiles/              # Deck structure profiles
│   │   ├── base_profile.py         # Abstract base profile
│   │   └── mewtwo_profile.py       # Mewtwo ex deck card mappings and constants
│   ├── expert_system/              # Expert coaching and heuristic knowledge
│   │   ├── base_expert.py          # Epoch schedule and board context extractor
│   │   ├── energy_evaluator.py     # Card energy requirement calculator
│   │   ├── expert_router.py        # Entropy and risk-adaptive confidence scaling
│   │   ├── mewtwo_expert.py        # Mewtwo-specific strategic rules and vetoes
│   │   └── search_strategy.py      # Tutor search prioritization algorithms
│   ├── reward_system/              # Granular RL reward shaping
│   │   ├── base_reward.py          # Domain-agnostic strategic transition rewards
│   │   ├── mewtwo_reward.py        # Mewtwo deck safety rewards and rule guards
│   │   └── reward_router.py        # Reward calculation router and clipping
│   ├── strategies/                 # Strategic documentation and matchup analysis
│   │   ├── intro.md                # Strategy index
│   │   ├── mewtwo_strategy.md      # Strategic guide for Mewtwo ex archetype
│   │   └── mewtwo_matchup_strategy.md # Detailed matchup breakdown against 50+ bots
│   ├── training/                   # Reinforcement learning components
│   │   ├── card_database.py        # Card and attack lookup tables
│   │   ├── checkpoint.py           # Atomic checkpoint manager
│   │   ├── evaluator.py            # Rule-based opponent loader, SPRT, and Elo
│   │   ├── gae.py                  # Generalized Advantage Estimation calculation
│   │   ├── inference_server.py     # Batched GPU inference server
│   │   ├── logger.py               # Progress bar and metrics CSV logging
│   │   ├── replay_buffer.py        # Prioritized experience replay buffer
│   │   ├── rewards.py              # Reward engine facade
│   │   └── worker.py               # Asynchronous self-play simulation worker
│   ├── agent.py                    # MCTS agent, Node expansion, and Kaggle entrypoint
│   ├── expert_knowledge.py         # Top-level facade for expert guidance
│   ├── model.py                    # Transformer Policy-Value Network (MyModel)
│   └── plot_metrics.py             # Matplotlib training visualization generator
├── .gitignore                      # Git exclusion rules (excludes weights and raw datasets)
├── best_model.pth                  # Best trained checkpoint (~306 MB, gitignored)
├── deck.csv                        # Active 60-card deck configuration
├── deck_mewtwo.csv                 # Mewtwo ex deck backup definition
├── EN_Card_Data.csv                # Card metadata catalog (360 KB)
├── extract_deck_visualizer.py      # Script to extract replay game logs for visualizer
├── main.py                         # Kaggle competition agent loader entrypoint
├── package_submission.py           # Automated bundle packager (outputs submission.tar.gz)
├── README.md                       # Comprehensive repository documentation
├── tcg_replay_viewer.py            # Local replay analysis script
├── test_agent.py                   # Agent and MCTS integration test
├── test_energy_block_unit.py       # Energy requirement unit tests
├── test_mewtwo_profile.py          # Mewtwo deck profile tests
├── test_reward_fixes.py            # Reward calculation unit tests
├── test_search_strategy.py         # Tutor search target selection tests
├── train.py                        # Master training script (PPO, self-play, SPRT)
├── utils.py                        # Shared utility functions
└── visualizer.html                 # Interactive browser-based match replay viewer
```

---

## 12. Installation

### Requirements
* **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
* **Python**: Version 3.10, 3.11, or 3.12 (Python 3.12.3 recommended)
* **PyTorch**: Version 2.0 or newer with CUDA support (for GPU training)
* **Hardware**:
  * **Inference**: Any modern CPU (sub-5ms per decision).
  * **Training**: 8+ CPU cores and an NVIDIA GPU with $\ge 8\text{ GB}$ VRAM recommended.

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/ChaiWork/PTCG-AI.git
cd PTCG-AI

# 2. Create and activate a Python virtual environment
# Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install torch numpy matplotlib pytest
```

> [!NOTE]
> The `cg/` directory contains pre-compiled C++ engine libraries for Windows (`cg.dll`), Linux (`libcg.so`), and macOS (`libcg.dylib`). No separate compilation or C++ build tools are required.

---

## 13. Usage

### 1. Training from Scratch or Checkpoint
Run the master training script with self-play and evaluation:

```bash
python train.py --epochs 25 --self-play-episodes 100 --eval-episodes 50 --batch-size 128 --lr 5e-5 --num-workers 4
```

* To resume training from an existing checkpoint while resetting the curriculum schedule:
  ```bash
  python train.py --reset-epoch --epochs 20 --self-play-episodes 100 --batch-size 128
  ```

### 2. Packaging for Kaggle Submission
Generate the `submission.tar.gz` package formatted for Kaggle upload:

```bash
python package_submission.py
```
This automatically verifies that `best_model.pth`, `main.py`, `deck.csv`, `src/`, and `cg/` are bundled at the root level and confirms the total archive size is below the 197.7 MiB limit.

### 3. Running Unit Tests
Execute the test suite using `pytest`:

```bash
pytest
```
Or test specific modules:
```bash
python -m unittest test_agent.py
python -m unittest test_mewtwo_profile.py
python -m unittest test_energy_block_unit.py
```

### 4. Match Replay Visualizer
To view and analyze match decisions in an interactive HTML interface:
```bash
python extract_deck_visualizer.py
```
Open [`visualizer.html`](file:///d:/codingProject/pokemon-tcg-ai-agent/pokemon-tcg-ai-battle-challenge-strategy/visualizer.html) in any modern web browser to step through card plays, energy attachments, and prize trades.

---

## 14. Configuration & Hyperparameters

Training and evaluation parameters are configurable via CLI arguments in `train.py`:

| Parameter | Flag | Default | Description |
|---|---|:---:|---|
| **Epochs** | `--epochs` | `5` | Number of training iterations |
| **Self-Play Episodes** | `--self-play-episodes`| `100` | Games collected per epoch across worker processes |
| **Evaluation Episodes**| `--eval-episodes` | `50` | Maximum games evaluated per epoch under SPRT |
| **Self-Play Ratio** | `--self-play-ratio` | `0.5` | Ratio of games against Current Self vs historical league |
| **Batch Size** | `--batch-size` | `128` | Training minibatch size |
| **Learning Rate** | `--lr` | `5e-5` | Peak AdamW learning rate |
| **Buffer Capacity** | `--buffer-size` | `20000` | Prioritized Replay Buffer maximum capacity |
| **Worker Processes** | `--num-workers` | `4` / `CPU-1`| Parallel simulation processes |
| **Early Stopping** | `--patience` | `10` | Consecutive rejected epochs before early stopping |
| **League Play** | `--disable-league` | `False` | Disables historical league checkpoint evaluation |
| **Epoch Reset** | `--reset-epoch` | `False` | Resets epoch counter to restore expert coaching schedule |

---

## 15. Model Checkpoints

Model weight files (`.pth`) are gitignored to avoid repository bloat (~306 MB each).

### Trained Checkpoints
* **`best_model.pth`**: Peak evaluation checkpoint (69.5%–70.5% win rate).
* **`latest_model.pth`**: Latest state checkpoint saved at each epoch.

### Downloading Pre-Trained Weights
Trained weights are hosted as assets on [GitHub Releases](https://github.com/ChaiWork/PTCG-AI/releases):

```bash
# Windows PowerShell:
Invoke-WebRequest -Uri "https://github.com/ChaiWork/PTCG-AI/releases/download/v1.0.0/best_model.pth" -OutFile "best_model.pth"

# Linux / macOS:
wget https://github.com/ChaiWork/PTCG-AI/releases/download/v1.0.0/best_model.pth
```
Place `best_model.pth` in the project root directory.

### Loading Checkpoints in Code
```python
import torch
from src.model import (
    MyModel, MODEL_D_MODEL, MODEL_NUM_HEADS,
    MODEL_D_FEEDFORWARD, MODEL_NUM_LAYERS_ENCODER, MODEL_NUM_LAYERS_DECODER,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(
    MODEL_D_MODEL, MODEL_NUM_HEADS, MODEL_D_FEEDFORWARD,
    MODEL_NUM_LAYERS_ENCODER, MODEL_NUM_LAYERS_DECODER,
).to(device)

checkpoint = torch.load("best_model.pth", map_location=device, weights_only=True)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)
model.eval()
```

---

## 16. Reproducibility

* **Deterministic Random Seeds**: `train.py` sets fixed seeds (`SEED = 42`) for `random`, `torch`, and `torch.cuda`.
* **Verified Runtime Environment**:
  * Python 3.12.3 (64-bit) on Windows 11 / Linux Ubuntu 22.04
  * PyTorch 2.4+ with CUDA 12.x
  * NumPy 1.26+
* **Training Time**: Approximately 15–20 minutes per epoch (100 self-play games + 50 SPRT evaluation games + 20 training minibatches) on an 8-core CPU with an NVIDIA RTX 3070/4070 GPU.

---

## 17. Limitations

1. **Information-State Determinization Approximation**:
   Sampling hidden opponent hands and prizes uniformly from unrevealed cards assumes an unbiased distribution. Against opponents with distinct card retention patterns, true Bayesian opponent modeling would provide superior inference.
2. **Single-Deck Optimization**:
   The current expert coaching and reward system are specialized for the Team Rocket Mewtwo ex control archetype. While base rewards are domain-agnostic, running other deck archetypes requires corresponding deck profiles.
3. **Discrete Action Space Constraints**:
   The decoder evaluates up to 64 candidate action combinations per step. Rare complex actions requiring combinations of 3+ simultaneous selections may be truncated.
4. **Adversarial Exploitation**:
   Opponents employing extreme stall tactics or non-standard card counts (e.g., pure energy stall) can occasionally force long games (25+ turns), though win rates remain positive.

---

## 18. Future Work

* **Dynamic Bayesian Opponent Hand Tracking**: Track discarded cards and played search items to compute exact probability distributions over hidden cards rather than uniform random sampling.
* **Multi-Deck Unified Agent**: Extend the policy-value network to condition on arbitrary active deck lists without archetype-specific reward shaping.
* **Asynchronous League Expansion**: Scale the league pool to include real-time tournament replay traces from top human players.
* **Sub-Tree Reuse**: Preserve relevant subtrees across sequential game turns in MCTS to reduce rollout computational overhead.

---

## 19. Academic & Competition Context

This project was developed for the **Kaggle Pokémon TCG AI Battle Challenge** as part of advanced research into reinforcement learning for imperfect-information games.

* **Core Methodology**: Adapts AlphaZero principles (MCTS + Deep Dual-Head Networks) to stochastic imperfect-information environments using information-state determinization and expert-guided PUCT exploration.
* **Empirical Validation**: Evaluated against 53 distinct opponent models representing diverse archetypes (Aggro, Spread, Control, Setup) and top Kaggle competitor approaches.

---

## 20. License

License information will be added.
