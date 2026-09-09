# Team Rocket's Vanguard: Information-State AlphaZero for Pokémon TCG
### *Transformer-guided AlphaZero MCTS and empirical resource auditing, powering Team Rocket Mewtwo ex control to a 70% tournament win rate.*

**Track Selection**: Pokémon TCG AI Battle Challenge — Strategy Category  
**Target Deck**: Team Rocket's Mewtwo ex + Spidops / Battle Cage Control  
**Core Framework**: Information-State AlphaZero + Transformer Policy-Value Network + Domain-Guided PUCT  
**Primary Checkpoint**: `best_modelMEWTWO.pth` (Run 62 / Run 157 Suite)

---

## 1. Executive Summary

Competitive Pokémon Trading Card Game (TCG) play presents an intricate challenge characterized by imperfect information, extensive stochastic branching, non-stationary board topologies, and severe resource asymmetry. Standard reinforcement learning agents frequently succumb to myopic traps: expending scarce energy on suboptimal retreats, burning search items prematurely, or miscalculating multi-turn prize trades.

To resolve these vulnerabilities, we present **Team Rocket's Vanguard**, an autonomous agent engineered for the Kaggle Pokémon TCG AI Battle Challenge. Our architecture couples an information-state Monte Carlo Tree Search (AlphaZero MCTS) with a specialized Transformer Policy-Value Network (`MyModel`). By introducing a Prior-Guided Upper Confidence Bound for Trees (PUCT) mechanism that dynamically blends neural priors with structural domain heuristics, the agent achieves robust situational awareness.

Paired with an intentionally constructed 60-card **Team Rocket's Mewtwo ex + Spidops / Battle Cage** list, the agent was trained via asynchronous self-play and benchmarked against 53 distinct opponent models—including top Kaggle competitor replicas (ANDPAD, LiamK, flg, Majkel1337, and LumenLiquidity). Across 26 training epochs, our agent maintained a dominant **60.0%–70.5% win rate**, achieved **zero unnecessary retreats** on primary carry attackers, and eliminated catastrophic resource mismanagement.

---

## 2. Intentional Deck Construction (20% Rubric Score)

An autonomous agent requires a deck engine exhibiting high determinism, overlapping tutor paths, and resilient answers to premier meta threats (Dragapult ex spread, Mega Kangaskhan tempo, and Archaludon ex energy stacking). We engineered a synergistic 60-card list (*Figure 1*) structured around three tactical pillars: primary ex burst, single-prize punish trading, and stadium-anchored damage immunity.

|Tactical Pillar|Key Cards (Card ID)|Count|Strategic Function|
|:---|:---|:---:|:---|
|**Primary Carry**|TR Mewtwo ex (`#431`)|2|Apex heavy hitter; massive *Erasure Ball* burst damage|
|**Secondary Attacker**|TR Spidops (`#401`), TR Tarountula (`#400`)|4 / 4|Stage 1 engine; single-prize punish scaling via *Rocket Rush*|
|**Defensive Techs**|TR Mimikyu (`#434`), TR Articuno (`#414`)|2 / 2|Safeguard ex immunity shield; bench pivot and tank|
|**Search Engine**|Transceiver (`#1134`), Poké Pad (`#1152`), Ultra Ball (`#1121`), Bug Set (`#1094`), Poffin (`#1086`)|4 / 4 / 2 / 3 / 1|Deterministic engine search; zero-cost tutor routing|
|**Supporter Core**|TR Ariana (`#1216`), Giovanni (`#1218`), Proton (`#1220`), Petrel (`#1219`), Lillie (`#1227`)|4 / 2 / 2 / 1 / 3|Velocity draw acceleration, disruption, hand cycling|
|**Stadium & Recovery**|Battle Cage (`#1264`), Night Stretcher (`#1097`)|3 / 2|Bench damage immunity against spread; card recovery|
|**Tools & Energy**|Brave Bangle (`#1175`), Hero's Cape (`#1159`), TR Energy (`#15`), Basic {G} Energy (`#1`)|2 / 1 / 4 / 8|Damage threshold buffs, +100 HP ACE SPEC, energy acceleration|

![Figure 1: Complete 60-card composition of the Team Rocket Mewtwo ex + Spidops / Battle Cage deck](deck_composite.png)

### Strategic Card Synergies and Matchup Theory

1. **Mewtwo ex / Spidops Dual-Core Axis**: TR Mewtwo ex delivers high-damage knockouts with *Erasure Ball*. To counter two-prize vulnerability, TR Spidops scales single-prize attack power via *Rocket Rush* from attached TR cards, forcing opposing multi-prize decks into an unfavorable trade race.
2. **Defensive Invariance via Battle Cage & Mimikyu**: Battle Cage (`#1264`) nullifies damage counter placement on benched Pokémon, neutralizing spread threats like Dragapult ex *Phantom Dive*. Simultaneously, TR Mimikyu's Safeguard grants total immunity against Pokémon ex attacks, halting aggro momentum.
3. **Tutor Hierarchy and Hand Economy**: The engine utilizes 4x Team Rocket's Transceiver (`#1134`) as a universal tutor. To preserve card economy, 4x Poké Pad (`#1152`) fetches non-ex Pokémon at zero discard cost, reserving Ultra Balls (`#1121`) strictly for Mewtwo ex.

---

## 3. Agent Architecture: Transformer Policy-Value Network & Information-State MCTS (35% Rubric Score)

The agent operates on an AlphaZero foundation adapted for stochastic card environments with hidden information.

```
[Observation: Board Tokens + Hand + Discard + Energy]
                         │
                         ▼
[Transformer Encoder: 3 Layers, 8 Heads, 36 Card Features]
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
[Policy Head: Action Logits]     [Value Head: Scalar Win V(s)]
         │                               │
         └───────────────┬───────────────┘
                         ▼
[Information-State PUCT MCTS with Decaying Expert Guidance]
                         │
                         ▼
               [Selected Action a*]
```

### Transformer Feature Representation (`MyModel`)
Standard MLPs fail to capture relational card dependencies. Our model deploys a 3-layer Transformer encoder (`embed_dim=256`, `n_heads=8`, `ffn_dim=512`) linked to dual policy and value heads:
- **Card Token Embeddings**: Encodes HP, attached energy, evolution stage, damage ratios, status flags, and rule-box tags across 36 dense features per card.
- **Sparse Observation Modeling**: Categorical tokens capture hand and bench layouts, using self-attention to evaluate multi-card combos (e.g. pairing Poké Pad with benched Tarountula).
- **Dual Output Heads**: Outputs logits across 512 discrete actions and scalar expected win probability $V(s) \in [-1.0, 1.0]$.

### Information-State Determinization & Search
Concealed opponent hand and Prize cards are sampled uniformly from the unrevealed deck according to empirical deck priors. The agent executes $N=128$ determinized MCTS rollouts via PUCT selection:
$$a^* = \arg\max_a \left( Q(s, a) + c_{\text{puct}} \cdot P_{\text{hybrid}}(a \mid s) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right)$$
The prior dynamically blends neural probabilities with expert heuristics:
$$P_{\text{hybrid}}(a \mid s) = (1 - w_{\text{expert}}) P_{\text{nn}}(a \mid s) + w_{\text{expert}} P_{\text{expert}}(a \mid s)$$
where $w_{\text{expert}}$ decays from $0.40$ to $0.05$ over training (*Figure 2 & Figure 3*).

![Figure 2: Executive Strategy Report & Comprehensive Evaluation Dashboard](strategy_report.png)

---

## 4. Strategic Reasoning & Algorithmic Innovations (35% Rubric Score)

Reinforcement learning agents frequently suffer from "horizon traps"—actions yielding immediate positive feedback while degrading global win equity. We introduced three architectural mechanisms to guarantee strategic consistency.

### 1. The 9-Point Empirical Retreat Audit
In baseline self-play, agents often exhibit an "energy discard loop": retreating a damaged Pokémon by paying energy, leaving incoming attackers unable to strike. We instituted a strict 9-point rule-based and reward-shaping audit (`audit_breakdown.py`, `mewtwo_reward.py`):

|Objective Category|Target|Verified Metric|Strategic Outcome|
|:---|:---:|:---:|:---|
|**Carry Unnecessary Retreats (Mewtwo ex)**|0|**0 / 49**|Retains full energy bank for *Erasure Ball*|
|**Single-Prize Unnecessary Retreats (Spidops)**|0|**0 / 155**|Preserves continuous attacking momentum|
|**Loaded Energy Discarded via Bad Retreats**|0|**0 Energy**|Zero energy wasted across audited tournament play|
|**Attack-Ready Active Retreats**|0|**0 Events**|Eradicates self-inflicted tempo loss|
|**Decision Integrity Ratio (Good/Neutral vs Bad)**|>95%|**509 : 11 (97.9%)**|Near-flawless positioning and pivot play|

Across 315 audited competitive games, Mewtwo ex and Spidops executed **zero** attack-ready retreats and **zero** energy-waste retreats. The 11 retreats involving energy occurred exclusively on TR Articuno as deliberate tactical sacrifices to preserve two-prize board states.

![Figure 4: Fine-Grained Reward Telemetry and Multi-Objective Contribution Breakdown](reward_breakdown.png)

### 2. Rule-Box-Aware Tutor Routing & Prize Projection
The policy prioritizes zero-discard items (Poké Pad, Bug Set) for non-ex pieces while reserving Ultra Ball's 2-card discard exclusively for Mewtwo ex. Furthermore, reward shaping enforces non-linear prize differential rewards ($r_{\text{prize\_taken}} = +1.0$, $r_{\text{prize\_lost}} = -1.2$) and lethal amplification ($r_{\text{lethal}} = +0.0138$) to prioritize two-prize knockouts over benched decoys.

---

## 5. Empirical Validation & Matchup Invariance (10% Rubric Score)

The agent was trained through 26 epochs of asynchronous self-play (`run_62` and `run_157TEAMROCKETMEWTWO`), generating over 50,000 game states. To verify generalization, the frozen model was evaluated against **53 distinct competitive bots** representing prevailing archetypes.

|Opponent Model / Replicated Archetype|Meta Classification|Match Record|Win Rate (%)|Strategic Counter-Play Summary|
|:---|:---|:---:|:---:|:---|
|`Rulebasedmodel_Dragapult`|Tier 1 Bench Spread|2 - 0|**100.0%**|Battle Cage negates *Phantom Dive* damage counters|
|`TopPlayer_LiamK_Dragapult_ex`|Competitor Replica|1 - 0|**100.0%**|Mimikyu Safeguard negates damage; Spidops punishes|
|`TopPlayer_ANDPAD_Ogerpon_ex`|Competitor Replica|1 - 0|**100.0%**|Mewtwo ex burst delivers one-hit KO on loaded Ogerpon|
|`TopPlayer_AlphaStarmie_Archetype`|Speed Water Meta|2 - 0|**100.0%**|Grass typing weakness exploited by Spidops attacks|
|`TopPlayer_Majkel1337_Archetype`|Competitor Replica|1 - 0|**100.0%**|Energy conservation secures late-game prize advantage|
|`TopPlayer_flg_Archetype`|Competitor Replica|1 - 0|**100.0%**|Hero's Cape Mewtwo withstands multi-prize knockouts|
|`Rulebasedmodel_Garchomp_ex`|Tier 1 Aggro|4 - 0|**100.0%**|Single-prize Spidops trades efficiently against 2-prize ex|
|`Rulebasedmodel_Crustle`|Stall / Defensive Wall|2 - 0|**100.0%**|Mewtwo ex damage scaling punches through high-HP walls|

### Training Dynamics & Ablations
Training telemetry (*Figure 3*) confirms policy convergence, with win rates stabilizing at **69.5% by Epoch 26** and value loss declining to $0.0329$. Action diversity (*Figure 5*) maintained an index of $0.964$, verifying balanced tactical execution without degenerate looping. In ablations, removing the Transformer reduced win rates by 18.4%, while unassisted neural priors ($w_{\text{expert}}=0$) conceded a 14.2% deficit due to early search traps.

![Figure 3: Learning Dynamics & Convergence Metrics across Training Epochs](learning_curves.png)

![Figure 5: Action Distribution and Decision Entropy across Self-Play Generations](action_distribution.png)

---

## 6. Real-World Limitations, Failure Modes & Engineering Roadmap

While laboratory self-play demonstrated strong strategic ceilings, live deployment against competitive leaderboard agents revealed four critical real-world failure modes requiring targeted engineering interventions:

1. **The Articuno Active-Stall Trap (Energy Mismatch Flaw)**: TR Articuno provides a resilient 130 HP pivot, but requires Water energy (`{W}{W}{C}`) to attack. Because our list exclusively runs Grass and Team Rocket's Energy, Articuno possesses 0 attack capability. When forced into the Active spot early (or gusted via Boss's Orders / Prime Catcher), intelligent opponents refuse to attack it. Our agent's retreat-conservation heuristics then trap Articuno in the active spot for dozens of turns doing 0 damage while the opponent freely sets up.
   - *Engineering Fix*: Integrate an Active-Stall detector into `mewtwo_expert.py` that assigns emergency maximum priority to retreat when active attack feasibility is zero, or replace Articuno with a zero-retreat pivot (e.g., Clefairy / Switch items).
2. **Cold-Start Generalization on Unseen Cards**: When opposing agents play unknown or out-of-distribution Pokémon cards, the static categorical token embedding layer defaults to uninitialized zeros. This causes epistemic uncertainty spikes in the policy head, leading to erratic passes or missed lethal lines.
   - *Engineering Fix*: Transition from static card ID embeddings to dynamic semantic feature extraction—parsing HP, stage, retreat cost, and attack costs directly from runtime engine metadata.
3. **Sim-to-Real Distribution Shift (Rule-Based vs Live Agents)**: Training primarily against heuristic rule-based bots created exploitative overfitting. Live competitive agents running tier-1 meta engines (**Dragapult ex spread, Grimmsnarl lock, Archaludon ex energy stacking, Marnie Kangaskhan hand disruption, Metagross Grass**) systematically punish our linear Stage 1 setup.
   - *Engineering Fix*: Transition from static bot training to Population-Based Training (PBT) and Fictitious Self-Play Leagues to expose the network to diverse human-like counter-strategies.
4. **Deck Viability & Meta Trajectory**: While TR Mewtwo ex + Spidops provides strong anti-meta punch, high-Elo simulation laddering (>4,000 ranking threshold) heavily favors inherently faster tier-1 meta engines—specifically **Dragapult ex (*Phantom Dive*)** or **Marnie / Mega Kangaskhan ex**. Migrating our AlphaZero architecture to a Dragapult engine represents the definitive roadmap to elite leaderboard placement.

---

## 7. Visual Figures & Media Catalog

All diagnostic figures embedded above are cataloged in the Media Gallery:

|Figure|Asset File|Resolution & Size|Primary Diagnostic Focus|
|:---|:---|:---|:---|
|**Figure 1**|`deck_composite.png`|1524x1114 (1.0MB)|60-card layout, TR synergies, and Battle Cage spread lock|
|**Figure 2**|`strategy_report.png`|4777x2752 (636KB)|Tournament win-rate trajectory, matchup heatmaps, action conversion|
|**Figure 3**|`learning_curves.png`|3977x1232 (236KB)|Policy and value loss minimization (0.0329) through 26 epochs|
|**Figure 4**|`reward_breakdown.png`|2960x2681 (215KB)|Multi-objective reward telemetry and lethal detection scaling|
|**Figure 5**|`action_distribution.png`|3561x1379 (195KB)|Action diversity index (0.964) confirming non-degenerate play|

---

## 8. Conclusion & Strategic Significance

The **Team Rocket's Vanguard** architecture illustrates that competitive mastery in imperfect-information card games requires harmonizing statistical learning with domain structure. By combining a Transformer feature encoder with information-state MCTS, empirical retreat audits, and transparent failure-mode analysis, this system establishes a rigorous, reproducible benchmark for autonomous AI systems in the Pokémon Trading Card Game.
