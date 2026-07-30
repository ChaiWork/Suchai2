import glob
import math
import os
import random
import sys
import torch
from collections import Counter

"""
RL Strategy: Pokémon TCG AI Agent
==================================
Primary Objective: Maximize long-term win rate against diverse deck archetypes.

Decision Priorities (encoded in reward shaping):
  1. Win the game           — terminal reward ±1.0
  2. Take prizes efficiently — r_prize_taken * 0.20 (key win condition)
  3. Prevent opponent setup — r_prize_lost * 0.15 (opponent taking prizes)
  4. Attach energy first    — r_energy * 0.03 (prerequisite to attacking)
  5. Maintain attackers     — bench reward (+0.10 recovery, -0.02 danger)
  6. Play efficiently       — r_stall -0.005/step (anti-stall pressure)

Tactical Action Sequence (optimal turn order):
  Abilities → Items/Search → Energy Attachment → Supporter → Attack

Key Principle: Every action must increase P(win), not just deal damage.
"""


from model import (
    MyModel,
    SparseVector,
    get_encoder_input,
    get_decoder_input,
    eval_nn,
    MODEL_D_MODEL,
    MODEL_NUM_HEADS,
    MODEL_D_FEEDFORWARD,
    MODEL_NUM_LAYERS_ENCODER,
    MODEL_NUM_LAYERS_DECODER,
)

try:
    from expert_knowledge import get_expert_bonus, EXPERT_WEIGHT, USE_EXPERT_GUIDANCE
except ImportError:
    from src.expert_knowledge import get_expert_bonus, EXPERT_WEIGHT, USE_EXPERT_GUIDANCE

# Resolve cg-lib path dynamically for Kaggle vs Local environments
try:
    cg_lib_path = glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0]
    sys.path.append(cg_lib_path)
except IndexError:
    pass

from cg.api import (
    SearchState,
    to_observation_class,
    search_begin,
    search_step,
    search_end,
    OptionType,
    SelectContext,
    CardType,
    AreaType,
    all_card_data,
)

card_table = {c.cardId: c for c in all_card_data()}

SEARCH_COUNT = 200  # MCTS Search count — ≥200 needed for meaningful visit differentiation (audit: was 50, policy targets were noise)
C_PUCT = 1.25       # AlphaZero PUCT exploration constant (fixed, per original paper)


class LearnSample:
    """Single Training Sample collected during gameplay/self-play."""
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec: SparseVector):
        self.value = value        # Encoder output (expected outcome)
        self.policy = policy      # Decoder output (action probability distribution)
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec


class GPUInferenceClient:
    """Client wrapper for workers to send inference queries to the GPU batch server."""
    def __init__(self, conn):
        self.conn = conn

    def eval_nn(self, sv_enc: SparseVector, sv_dec: SparseVector) -> tuple[float, list[float]]:
        try:
            self.conn.send((sv_enc, sv_dec))
            return self.conn.recv()
        except Exception:
            num_actions = len(getattr(sv_dec, "offset", []))
            if num_actions <= 0:
                num_actions = 1
            uniform_prob = 1.0 / num_actions
            return 0.0, [uniform_prob] * num_actions



class Child:
    """Edge in the MCTS search tree pointing to a child node."""
    def __init__(self, select: list[int], prob: float, select_option=None):
        self.node = None
        self.select = select
        self.prob = prob
        self.select_option = select_option


class Node:
    """MCTS Node representing a game state in the search tree."""
    value: float          # Self value
    total: float          # Total accumulated value
    visit: int            # Visit count
    parent: 'Node | None' # Parent node
    children: list[Child]
    state: SearchState    # Search State of this node

    def __init__(self, parent: 'Node | None', state: SearchState):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    def backprop(self, value: float):
        """Backpropagate the value up the search tree."""
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


def detect_opponent_archetype(obs, your_index: int) -> str:
    """Detect opponent deck archetype from revealed active, bench, and discard cards."""
    if obs is None or obs.current is None:
        return "GENERIC"
    try:
        opp_index = 1 - your_index
        opp_player = obs.current.players[opp_index]
        visible_cards = []
        for card in opp_player.active + opp_player.bench + opp_player.discard:
            name = getattr(card, 'name', '').lower()
            visible_cards.append(name)
        card_str = " ".join(visible_cards)
        if "dreepy" in card_str or "drakloak" in card_str or "dragapult" in card_str:
            return "DRAGAPULT"
        elif "snover" in card_str or "abomasnow" in card_str:
            return "ABOMASNOW"
        elif "iono" in card_str or "pidgeot" in card_str or "snorlax" in card_str:
            return "IONO_CONTROL"
    except Exception:
        pass
    return "GENERIC"


def get_target_card_priority_score(opt, obs, archetype: str) -> float:
    """Calculate strategic priority score for targeting opponent cards based on matchup prompt."""
    if opt.type != OptionType.CARD:
        return 1.0
    
    card_name = ""
    try:
        if hasattr(opt, 'area') and hasattr(opt, 'index') and hasattr(opt, 'playerIndex'):
            target_player = obs.current.players[opt.playerIndex]
            if opt.area == AreaType.ACTIVE and target_player.active:
                card_name = getattr(target_player.active[0], 'name', '').lower()
            elif opt.area == AreaType.BENCH and opt.index < len(target_player.bench):
                card_name = getattr(target_player.bench[opt.index], 'name', '').lower()
    except Exception:
        pass

    if not card_name:
        return 1.0

    if archetype == "DRAGAPULT":
        if "dreepy" in card_name:
            return 10.0   # Highest Priority 1: KO Dreepy immediately
        elif "drakloak" in card_name:
            return 8.0    # Priority 2: KO Drakloak before Dragapult evolves
        elif "dragapult" in card_name:
            return 5.0    # Priority 3: Dragapult ex
    elif archetype == "ABOMASNOW":
        if "snover" in card_name:
            return 10.0   # Highest Priority 1: KO Snover immediately
        elif "abomasnow" in card_name:
            return 7.0    # Priority 2: Abomasnow
    elif archetype == "IONO_CONTROL":
        if "iono" in card_name or "pidgeot" in card_name or "bibarel" in card_name:
            return 8.0    # Draw/Support engine
            
    # Universal Rule: Target evolving pre-evolutions (Basic Pokémon) over fully evolved forms
    return 3.0


def create_node(parent: Node | None,
                search_state: SearchState,
                your_index: int,
                your_deck: list[int],
                model: MyModel,
                epoch: int = 1
) -> tuple[Node, LearnSample | None]:
    """Create a new node and evaluate its state using the neural network."""
    node = Node(parent, search_state)
    obs = search_state.observation
    state = obs.current
    
    if state.result >= 0:
        # Battle finished
        if state.result == 2:
            node.value = 0.0
        elif state.result == your_index:
            node.value = 1.0
        else:
            node.value = -1.0
        # Backpropagation is handled exclusively during tree expansion in mcts_agent
        sample = None
    else:
        # Enumerate up to 128 potential action combinations prioritizing high-value options
        actions = []
        options = obs.select.option
        
        archetype = detect_opponent_archetype(obs, your_index)
        
        # High priority option types to guarantee evaluation
        HIGH_PRIORITY_TYPES = {
            OptionType.ATTACK,
            OptionType.EVOLVE,
            OptionType.ABILITY,
            OptionType.ATTACH,
            OptionType.RETREAT,
            OptionType.PLAY,
            OptionType.SPECIAL_CONDITION,
            OptionType.YES,
            OptionType.NO,
            OptionType.NUMBER,
            OptionType.SKILL,
            OptionType.END
        }
        
        scored_options = []
        for idx, opt in enumerate(options):
            score = 1.0
            if opt.type in HIGH_PRIORITY_TYPES:
                score += 2.0
            
            # Smooth Decision Hierarchy
            if opt.type == OptionType.ATTACK:
                score += 5.0
            elif opt.type == OptionType.EVOLVE:
                score += 4.0
            elif opt.type == OptionType.ABILITY:
                score += 4.0
            elif opt.type == OptionType.ATTACH:
                score += 3.0
            elif opt.type in (OptionType.PLAY, OptionType.CARD):
                score += 3.0
                try:
                    played_cid = getattr(opt, "cardId", -1)
                    if played_cid == -1 and hasattr(opt, "index") and obs.current:
                        hand = obs.current.players[your_index].hand
                        if 0 <= opt.index < len(hand) and hand[opt.index]:
                            played_cid = hand[opt.index].id
                    
                    played_card = card_table.get(played_cid)
                    bench_len = len(obs.current.players[your_index].bench) if (obs.current and len(obs.current.players) > your_index) else 0
                    if bench_len == 0 and played_cid in (400, 414, 434, 464, 272, 1086, 1121, 1134, 1094, 1092, 1216, 1220):
                        score += 10.0  # Top candidate priority for bench protection when bench is empty

                    if played_card:
                        # Generic Feature 1: High priority for ex attackers early
                        if getattr(played_card, "ex", False):
                            score += 5.0
                        # Generic Feature 2: Hand recovery for Supporters/Stadiums/Items when low on cards
                        hand_len = len(obs.current.players[your_index].hand) if obs.current else 5
                        if hand_len <= 2 and played_card.cardType in (CardType.SUPPORTER, CardType.STADIUM, CardType.ITEM):
                            score += 4.0
                        # Generic Feature 3: Tool attachment survivability
                        if played_card.cardType == CardType.TOOL:
                            score += 3.0
                    
                    # Supporter Contingency: Only play damage-buff supporters if active has energy to attack
                    if played_card and played_card.cardType == CardType.SUPPORTER and getattr(played_card, "damageBuff", False):
                        active_en = len(obs.current.players[your_index].active[0].energyCards) if (obs.current and obs.current.players[your_index].active) else 0
                        if active_en == 0:
                            score -= 5.0  # Soft penalty for playing damage buff without energy
                except Exception:
                    pass
                
            # Active Spot Promotion Safeguards against Heavy Threats (e.g. Mega Abomasnow ex)
            if obs.select and obs.select.context == SelectContext.TO_ACTIVE:
                try:
                    cand_cid = getattr(opt, "cardId", -1)
                    if cand_cid == -1 and hasattr(opt, "index") and obs.current:
                        players = obs.current.players[your_index]
                        if 0 <= opt.index < len(players.bench) and players.bench[opt.index]:
                            cand_cid = players.bench[opt.index].id
                    
                    opp_active = obs.current.players[1 - your_index].active
                    opp_is_heavy = False
                    if len(opp_active) > 0 and opp_active[0]:
                        opp_is_heavy = getattr(opp_active[0], "maxHp", 0) >= 200

                    cand_card = card_table.get(cand_cid)
                    if cand_card and not getattr(cand_card, "ex", False) and getattr(cand_card, "hp", 150) <= 100 and opp_is_heavy:
                        # Avoid floating low-HP non-ex Pokémon against heavy 200+ HP threat
                        score -= 10.0
                    elif cand_card and (getattr(cand_card, "ex", False) or getattr(cand_card, "stage1", False) or getattr(cand_card, "stage2", False)):
                        score += 3.0
                except Exception:
                    pass
            
            scored_options.append((idx, score))
            
        scored_options.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in scored_options]
        n = len(sorted_indices)
        k = obs.select.maxCount
        
        # Generate combinations using sorted_indices
        if k <= n:
            comb_positions = list(range(k))
            for _ in range(256):
                # Map combination positions to sorted_indices
                actions.append([sorted_indices[p] for p in comb_positions])
                
                # Standard next combination algorithm on positions [0, n-1]
                for i in range(k):
                    index = k - i - 1
                    if comb_positions[index] < n - i - 1:
                        comb_positions[index] += 1
                        for j in range(index + 1, k):
                            comb_positions[j] = comb_positions[j - 1] + 1
                        break
                else:
                    break
                    
        # Instrumentation: check if any options were truncated
        if len(options) > 256 and len(actions) == 256:
            included_indices = set()
            for act in actions:
                for idx in act:
                    included_indices.add(idx)
            dropped_types = set()
            for idx in range(len(options)):
                if idx not in included_indices:
                    dropped_types.add(options[idx].type)
            if dropped_types:
                dropped_type_names = []
                for t in dropped_types:
                    try:
                        dropped_type_names.append(OptionType(t).name)
                    except Exception:
                        dropped_type_names.append(str(t))
                try:
                    os.makedirs("out", exist_ok=True)
                    with open("out/truncated_options.log", "a", encoding="utf-8") as log_f:
                        ctx_name = obs.select.context.name if hasattr(obs.select.context, 'name') else str(obs.select.context)
                        log_f.write(f"Total: {len(options)}, Context: {ctx_name}, Dropped types: {dropped_type_names}\n")
                except Exception:
                    pass
                print(f"[DEBUG] Truncated options. Total: {len(options)}. Dropped OptionTypes: {dropped_type_names}", file=sys.stderr)
            
        sv_enc = get_encoder_input(obs, your_deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn(sv_enc, sv_dec, model)
        v = value
        if state.yourIndex != your_index:
            v = -v
        node.value = v
        # Backpropagation is handled exclusively during tree expansion in mcts_agent

        # Apply a prior bias to guide MCTS exploration towards constructive actions
        has_constructive = False
        for opt in options:
            if opt.type in [OptionType.ATTACK, OptionType.ATTACH, OptionType.EVOLVE, OptionType.PLAY, OptionType.ABILITY]:
                has_constructive = True
                break

        policy_biased = list(policy[:len(actions)])
        for i in range(len(actions)):
            bias = 0.0
            has_attack = False
            has_attach = False
            has_evolve = False
            has_play = False
            has_ability = False
            has_end = False
            
            for opt_idx in actions[i]:
                if opt_idx < len(options):
                    opt = options[opt_idx]
                    if opt.type == OptionType.ATTACK:
                        has_attack = True
                    elif opt.type == OptionType.ATTACH:
                        has_attach = True
                    elif opt.type == OptionType.EVOLVE:
                        has_evolve = True
                    elif opt.type == OptionType.PLAY:
                        has_play = True
                    elif opt.type == OptionType.ABILITY:
                        has_ability = True
                    elif opt.type == OptionType.END:
                        has_end = True
            
            bench_size_curr = len(obs.current.players[your_index].bench) if (obs.current and len(obs.current.players) > your_index) else 0
            has_setup_actions = has_play or has_attach or has_evolve or has_ability
            if has_play:
                bias += 2.5 if bench_size_curr == 0 else 1.2  # Massive priority prior to deploy bench protection when bench is empty
            if has_evolve:
                bias += 1.0
            if has_attach:
                bias += 0.8
            if has_ability:
                bias += 1.0  # High priority prior for activating Pokemon abilities before attacking
            if has_attack:
                if has_setup_actions:
                    bias += 0.5  # Lower attack bias while setup actions remain so Trainers are used first
                else:
                    bias += 1.5  # High attack bias when no setup actions remain
            if has_end and has_constructive:
                bias -= 5.0  # Penalize passing turn if constructive actions are possible
                
            prior_scale = 1.0 / math.sqrt(max(1, epoch))
            policy_biased[i] += bias * prior_scale

        # Convert raw policy logits to probabilities via numerically stable softmax.
        n_actions = len(actions)
        max_logit = max(policy_biased) if policy_biased else 0.0
        prob_sum = 0.0
        for i in range(n_actions):
            p = math.exp(policy_biased[i] - max_logit)
            first_opt = options[actions[i][0]] if (options and actions[i] and actions[i][0] < len(options)) else None
            node.children.append(Child(actions[i], p, select_option=first_opt))
            prob_sum += p
        if prob_sum > 0.0:
            for c in node.children:
                c.prob /= prob_sum
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)

# --- Opponent Deck Database & Belief State Identification ---
OPPONENT_DECKS = {
    'Rulebasedmodel': [673, 673, 674, 674, 675, 675, 676, 676, 676, 677, 677, 677, 678, 678, 678, 678, 1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141, 1141, 1141, 1142, 1142, 1142, 1142, 1152, 1152, 1152, 1152, 1159, 1182, 1182, 1192, 1192, 1192, 1192, 1227, 1227, 1227, 1227, 1252, 1252, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    'Rulebasedmodel_Abomasnow': [721, 721, 722, 722, 722, 722, 723, 723, 723, 723, 1121, 1121, 1121, 1121, 1126, 1192, 1192, 1192, 1192, 1227, 1227, 1227, 1227, 1262, 1262, 1262, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'Rulebasedmodel_Crustle': [1123, 344, 1123, 1, 1227, 1182, 1, 1, 1152, 345, 5, 1192, 1091, 1192, 1097, 158, 157, 1, 1182, 1, 344, 1, 1, 1192, 1227, 1227, 1, 344, 345, 7, 1086, 345, 1152, 345, 1182, 1097, 1159, 1227, 1, 5, 1, 158, 158, 1235, 1152, 1235, 6, 344, 1, 343, 1091, 1235, 1192, 343, 2, 1152, 1, 1231, 1, 1086],
    'Rulebasedmodel_Dipplin': [1086, 1097, 45, 1191, 1245, 93, 1086, 74, 1122, 93, 42, 1227, 93, 1, 90, 1245, 1227, 90, 1174, 1122, 89, 1094, 1231, 89, 1094, 42, 1, 1, 1245, 1, 1122, 89, 42, 1094, 1191, 100, 1184, 1129, 1227, 93, 42, 90, 1086, 1182, 1182, 1086, 1245, 73, 1080, 1123, 1122, 1094, 1227, 1211, 858, 240, 89, 1, 90, 1184],
    'Rulebasedmodel_Dragapult': [119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 140, 184, 235, 235, 1071, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1120, 1120, 1120, 1120, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1156, 1182, 1182, 1182, 1198, 1198, 1198, 1198, 1210, 1210, 1227, 1227, 1227, 1227, 1256, 1256, 2, 2, 2, 2, 5, 5, 5, 5],
    'Rulebasedmodel_Iono': [265, 265, 265, 268, 268, 268, 269, 269, 269, 270, 270, 270, 271, 271, 271, 1086, 1086, 1086, 1097, 1097, 1110, 1118, 1121, 1121, 1121, 1152, 1152, 1227, 1227, 1227, 1227, 1233, 1233, 1233, 1233, 1254, 1254, 1254, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    'Rulebasedmodel_Lucario': [673, 673, 674, 674, 675, 675, 676, 676, 676, 677, 677, 677, 678, 678, 678, 678, 1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141, 1141, 1141, 1142, 1142, 1142, 1142, 1152, 1152, 1152, 1152, 1159, 1182, 1182, 1192, 1192, 1192, 1192, 1227, 1227, 1227, 1227, 1252, 1252, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    'Rulebasedmodel_Mewtwo': [431, 431, 414, 414, 434, 434, 272, 401, 401, 401, 401, 400, 400, 400, 400, 1094, 1227, 1227, 1217, 1217, 1218, 1218, 1218, 1121, 1158, 1220, 1220, 1220, 1152, 1152, 1257, 1257, 1257, 1129, 1134, 1134, 1097, 1097, 1116, 1216, 1216, 1216, 1216, 1175, 1219, 1121, 1121, 1121, 1134, 1134, 5, 5, 1, 1, 1, 1, 1, 15, 15, 15],
    'Rulebasedmodel_Mewtwo_Easy': [431, 431, 414, 414, 434, 434, 272, 401, 401, 401, 401, 400, 400, 400, 400, 1094, 1227, 1227, 1217, 1217, 1218, 1218, 1218, 1121, 1158, 1220, 1220, 1220, 1152, 1152, 1257, 1257, 1257, 1129, 1134, 1134, 1097, 1097, 1116, 1216, 1216, 1216, 1216, 1175, 1219, 1121, 1121, 1121, 1134, 1134, 5, 5, 1, 1, 1, 1, 1, 15, 15, 15],
    'Rulebasedmodel_Mewtwo_Wobbuffet': [1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 15, 15, 15, 15, 400, 400, 400, 400, 401, 401, 401, 401, 414, 414, 431, 431, 432, 1094, 1094, 1094, 1097, 1119, 1119, 1134, 1134, 1134, 1134, 1152, 1152, 1152, 1152, 1159, 1175, 1216, 1216, 1216, 1216, 1217, 1218, 1218, 1218, 1219, 1220, 1220, 1227, 1227, 1257, 1257, 1257],
    'Rulebasedmodel_Starmie': [1227, 1227, 1225, 7, 1198, 1260, 104, 1030, 1145, 112, 7, 1152, 1182, 1097, 3, 3, 1225, 860, 1086, 1122, 1227, 1260, 3, 1152, 1031, 1030, 1152, 1225, 3, 1198, 112, 1145, 1086, 1174, 1086, 1159, 1097, 7, 1086, 1182, 1122, 1213, 1145, 1174, 1030, 1122, 112, 3, 1031, 1152, 860, 1260, 7, 1227, 1031, 860, 1229, 104, 1030, 861],
    'Rulebasedmodel_Kangaskhan_Crustle': [1, 11, 11, 11, 11, 14, 14, 14, 14, 18, 18, 18, 18, 343, 344, 344, 344, 344, 345, 345, 345, 345, 756, 756, 756, 756, 1086, 1086, 1086, 1086, 1087, 1122, 1122, 1122, 1122, 1123, 1123, 1123, 1123, 1147, 1147, 1147, 1147, 1159, 1182, 1182, 1197, 1197, 1197, 1197, 1225, 1225, 1225, 1225, 1227, 1227, 1227, 1227, 1264, 1264],
    'Rulebasedmodel_Grimmsnarl': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 104, 104, 112, 112, 112, 112, 646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 860, 860, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1152, 1152, 1152, 1152, 1122, 1137, 1182, 1182, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1231, 1259, 1259, 1259, 1259],
    'Rulebasedmodel_Trevenant': [879, 879, 879, 184, 311, 44, 878, 878, 878, 878, 140, 272, 343, 304, 858, 299, 1080, 1171, 1171, 1171, 1171, 1115, 1115, 1115, 1122, 1122, 1152, 1152, 1152, 1152, 1097, 1097, 1193, 1193, 1225, 1225, 1255, 1255, 1255, 1255, 1194, 1213, 1227, 1227, 1227, 1227, 1123, 1121, 1121, 1182, 1182, 1182, 19, 19, 19, 19, 11, 11, 11, 11],
    'Rulebasedmodel_Hydrapple_Ogerpon': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 93, 93, 96, 96, 96, 96, 140, 149, 149, 150, 150, 655, 709, 710, 710, 917, 917, 918, 920, 1071, 1071, 1079, 1088, 1094, 1094, 1094, 1094, 1097, 1121, 1121, 1121, 1152, 1182, 1182, 1184, 1188, 1201, 1227, 1227, 1227, 1227, 1231, 1261, 1261, 1261, 1261],
    'Rulebasedmodel_Lopunny': [11, 11, 11, 11, 13, 14, 14, 14, 65, 65, 65, 65, 66, 66, 66, 66, 848, 848, 848, 848, 849, 849, 849, 1086, 1086, 1086, 1086, 1102, 1102, 1102, 1102, 1122, 1122, 1122, 1122, 1152, 1152, 1152, 1152, 1174, 1174, 1174, 1174, 1182, 1225, 1225, 1225, 1225, 1227, 1227, 1227, 1227, 1229, 1229, 1229, 1229, 1264, 1264, 1264, 1264],
    'Rulebasedmodel_Typhlosion': [2, 2, 2, 2, 2, 119, 119, 119, 119, 120, 120, 120, 120, 140, 352, 352, 352, 352, 353, 353, 353, 353, 354, 354, 354, 354, 1079, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1114, 1122, 1122, 1122, 1122, 1129, 1152, 1152, 1152, 1152, 1184, 1213, 1213, 1213, 1213, 1215, 1215, 1215, 1215, 1227, 1227, 1227, 1227],
    'Rulebasedmodel_Grimmsnarl_ex': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 104, 104, 112, 112, 112, 112, 646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 860, 860, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1122, 1137, 1152, 1152, 1152, 1152, 1182, 1182, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1231, 1259, 1259, 1259, 1259],
    'Rulebasedmodel_Garchomp_ex': [6, 6, 6, 6, 6, 20, 20, 20, 117, 341, 341, 341, 341, 342, 342, 342, 379, 379, 379, 379, 380, 380, 380, 380, 381, 381, 381, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1122, 1122, 1141, 1141, 1142, 1142, 1142, 1152, 1152, 1152, 1152, 1173, 1173, 1182, 1182, 1182, 1182, 1213, 1213, 1227, 1227, 1227, 1227, 1256, 1261],
    'Rulebasedmodel_Marnie_Kangaskhan': [1, 11, 11, 11, 11, 14, 14, 14, 14, 18, 18, 18, 18, 344, 344, 344, 345, 345, 345, 756, 756, 756, 756, 1086, 1086, 1087, 1121, 1121, 1122, 1122, 1122, 1123, 1123, 1147, 1147, 1147, 1147, 1159, 1161, 1182, 1182, 1182, 1182, 1186, 1186, 1190, 1197, 1204, 1219, 1219, 1219, 1219, 1225, 1225, 1227, 1227, 1227, 1227, 1242, 1257],
    'Rulebasedmodel_Hydrapple_ex': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 92, 92, 96, 96, 96, 96, 140, 150, 150, 347, 347, 655, 655, 708, 708, 709, 709, 710, 710, 920, 1071, 1080, 1094, 1094, 1094, 1094, 1097, 1097, 1121, 1121, 1121, 1152, 1152, 1182, 1182, 1184, 1227, 1227, 1227, 1227, 1231, 1261, 1261, 1261],
    'Rulebasedmodel_Ogerpon_ex': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 42, 42, 42, 42, 96, 96, 96, 150, 150, 150, 347, 347, 347, 666, 666, 666, 666, 1079, 1079, 1079, 1079, 1086, 1086, 1094, 1094, 1094, 1094, 1122, 1122, 1122, 1152, 1152, 1159, 1224, 1224, 1224, 1231, 1231, 1231, 1231],
    'Rulebasedmodel_Garchomp_ex_2': [6, 6, 6, 6, 6, 20, 20, 20, 117, 341, 341, 341, 341, 342, 342, 342, 379, 379, 379, 379, 380, 380, 380, 380, 381, 381, 381, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1122, 1122, 1141, 1141, 1142, 1142, 1142, 1152, 1152, 1152, 1152, 1173, 1173, 1182, 1182, 1182, 1182, 1213, 1213, 1227, 1227, 1227, 1227, 1256, 1261],
    'Rulebasedmodel_HoOh_HeartGold': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 46, 46, 46, 46, 140, 140, 140, 357, 357, 357, 357, 855, 855, 1097, 1097, 1097, 1118, 1118, 1118, 1121, 1121, 1121, 1121, 1122, 1122, 1123, 1123, 1123, 1123, 1182, 1182, 1182, 1182, 1192, 1192, 1215, 1215, 1215, 1224, 1224, 1224, 1224, 1232, 1232, 1236, 1236],
    'Rulebasedmodel_Starmie_ex_2': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 721, 721, 721, 721, 1030, 1030, 1030, 1030, 1031, 1031, 1031, 1031, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1097, 1122, 1122, 1122, 1145, 1145, 1145, 1145, 1152, 1152, 1152, 1152, 1158, 1205, 1205, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235, 1235],
    'Rulebasedmodel_Metagross_Grass': [5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 272, 272, 272, 272, 547, 547, 639, 639, 639, 639, 640, 640, 641, 641, 641, 804, 804, 835, 835, 835, 1079, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1102, 1102, 1102, 1102, 1123, 1210, 1210, 1210, 1210, 1219, 1219, 1227, 1227, 1227, 1227],
    'Rulebasedmodel_Honchkrow': [15, 15, 15, 15, 17, 17, 17, 17, 414, 414, 463, 463, 463, 463, 473, 473, 474, 891, 891, 891, 891, 1097, 1097, 1109, 1122, 1122, 1122, 1122, 1134, 1134, 1134, 1134, 1152, 1152, 1152, 1152, 1216, 1216, 1216, 1216, 1217, 1217, 1217, 1217, 1218, 1218, 1218, 1218, 1219, 1219, 1219, 1219, 1220, 1220, 1220, 1220, 1257, 1257, 1257, 1257],
    'BasicallyBot_85134910': [788, 788, 788, 788, 789, 789, 789, 789, 928, 928, 928, 928, 855, 855, 855, 855, 1079, 1079, 1079, 1079, 1121, 1121, 1121, 1121, 1232, 1232, 1232, 1232, 1225, 1225, 1225, 1231, 1231, 1231, 17, 17, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'Cada_85134382': [119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 140, 184, 235, 1120, 1071, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 131, 131, 132, 133, 1121, 1121, 1121, 1120, 1120, 1152, 1152, 1152, 1182, 1182, 1182, 1198, 1198, 1198, 1198, 1210, 1210, 1227, 1227, 1227, 1227, 1256, 1256, 2, 2, 2, 2, 5, 5, 5, 5],
    'Cotini_85137077': [119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 131, 131, 132, 132, 133, 235, 140, 1071, 112, 1227, 1227, 1227, 1227, 1198, 1198, 1198, 1182, 1182, 1231, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1152, 1086, 1086, 1086, 1086, 1120, 1120, 1120, 1120, 1097, 1097, 1080, 1256, 1256, 1161, 343, 2, 2, 2, 5, 5, 5, 7, 7],
    'DarkLayer_85136498': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 57, 169, 169, 169, 169, 190, 190, 190, 190, 666, 666, 666, 666, 1097, 1097, 1097, 1121, 1121, 1121, 1121, 1122, 1122, 1122, 1122, 1147, 1147, 1147, 1147, 1152, 1152, 1152, 1152, 1159, 1182, 1182, 1182, 1185, 1185, 1185, 1185, 1213, 1227, 1227, 1227, 1227, 1244, 1244, 1244, 1244],
    'DRAGOPULT': [119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 131, 131, 132, 133, 112, 140, 184, 1086, 1086, 1086, 1086, 1121, 1121, 1121, 1121, 1079, 1079, 1097, 1097, 1182, 1182, 1182, 1227, 1227, 1227, 1198, 1198, 1201, 1240, 1231, 1152, 1152, 1152, 1152, 1080, 1246, 1246, 5, 5, 5, 5, 2, 2, 2, 7, 1079, 1123, 1119, 1122],
    'ek': [1158, 721, 721, 722, 722, 722, 722, 723, 723, 723, 723, 1145, 1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235, 1235, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'Gholdengo Lunatone': [186, 186, 186, 186, 191, 676, 676, 675, 675, 311, 547, 140, 695, 1182, 1182, 1182, 1184, 1142, 1142, 1142, 1086, 1088, 1174, 1174, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 700, 700, 700, 1213, 1213, 1213, 1213, 1123, 1123, 1118, 1118, 1118, 1118, 1121, 1121, 1121, 1121, 1119, 1119, 1097, 1171, 1177, 1086, 1086],
    'Hermes Lu': [119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 140, 184, 235, 235, 1071, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1120, 1120, 1120, 1120, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1156, 1182, 1182, 1182, 1198, 1198, 1198, 1198, 1210, 1210, 1227, 1227, 1227, 1227, 1256, 1256, 2, 2, 2, 2, 5, 5, 5, 5],
    'itofuki': [65, 65, 65, 66, 66, 66, 878, 878, 878, 878, 879, 879, 879, 879, 304, 304, 1227, 1227, 1227, 1227, 1182, 1182, 1182, 1213, 1213, 1086, 1086, 1086, 1086, 1152, 1152, 1152, 1152, 1097, 1097, 1097, 1097, 1080, 1123, 1123, 1115, 1115, 1122, 1122, 1171, 1171, 1171, 1171, 1255, 1255, 1255, 1255, 19, 19, 19, 19, 11, 11, 11, 11],
    'JBMetrix': [1030, 1030, 1030, 1030, 1031, 1031, 1031, 1031, 726, 726, 727, 727, 728, 478, 1086, 1086, 1086, 1086, 1145, 1145, 1145, 1145, 1227, 1227, 1227, 1227, 1182, 1182, 1205, 1205, 1122, 1122, 1122, 1122, 1097, 1097, 1123, 1123, 1235, 1235, 1213, 1158, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'Penguin': [756, 756, 756, 756, 344, 344, 344, 345, 345, 345, 1227, 1227, 1227, 1227, 1182, 1182, 1182, 1182, 1219, 1219, 1219, 1219, 1225, 1225, 1225, 1186, 1186, 1197, 1204, 1147, 1147, 1147, 1147, 1122, 1122, 1122, 1086, 1086, 1121, 1123, 1087, 1159, 1161, 1257, 1242, 1123, 14, 14, 14, 14, 18, 18, 18, 18, 11, 11, 11, 11, 1, 1086],
    'Raging Bolt Ogerpon': [172, 172, 172, 173, 173, 173, 63, 63, 96, 96, 174, 174, 171, 184, 140, 176, 75, 209, 1182, 1198, 1201, 1213, 1121, 1121, 1121, 1121, 1097, 1097, 1097, 1097, 1098, 1088, 1250, 1250, 1250, 1, 1, 1, 1, 1, 6, 6, 6, 4, 4, 4, 478, 171, 1198, 1198, 1198, 1132, 1132, 1132, 1132, 1119, 1119, 1119, 1124, 1152],
    'Ryan Talcoff': [149, 149, 149, 149, 93, 93, 150, 150, 150, 96, 96, 96, 920, 920, 920, 1079, 1079, 1079, 1079, 1086, 1086, 1086, 1086, 1121, 1121, 1121, 1121, 1152, 1152, 1227, 1227, 1227, 1227, 1192, 1192, 1182, 1182, 1182, 1123, 1123, 1116, 1116, 1097, 1097, 1175, 1175, 1158, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'Saikattyo_85136004': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11, 11, 11, 11, 14, 14, 14, 14, 18, 18, 18, 18, 344, 344, 344, 344, 345, 345, 345, 345, 1086, 1086, 1086, 1086, 1147, 1147, 1147, 1147, 1159, 1212, 1212, 1212, 1212, 1227, 1227, 1227, 1227, 1235, 1235, 1235, 1235],
    'Terapagos Noctowl': [176, 176, 176, 172, 172, 172, 173, 173, 173, 175, 175, 65, 65, 66, 66, 140, 1086, 1086, 1086, 1086, 1121, 1121, 1121, 1121, 1097, 1097, 1097, 1152, 1152, 1152, 1152, 1082, 1182, 1182, 1182, 1227, 1227, 1227, 1227, 1250, 1250, 1250, 1246, 1246, 1123, 1123, 1123, 1123, 1122, 1122, 1, 1, 1, 3, 3, 3, 4, 4, 4, 4],
}


def get_opponent_revealed_card_ids(obs, opponent_index: int) -> list[int]:
    """Scans visible zones (active, bench, discard) to find cards played by the opponent."""
    revealed = []
    ps = obs.current.players[opponent_index]
    
    # Active Pokémon + attached cards (energies, tools)
    for poke in ps.active:
        if poke is not None:
            revealed.append(poke.id)
            if poke.tools:
                revealed.extend(t.id for t in poke.tools if t)
            if poke.energyCards:
                revealed.extend(e.id for e in poke.energyCards if e)
                
    # Bench Pokémon + attached cards
    for poke in ps.bench:
        if poke is not None:
            revealed.append(poke.id)
            if poke.tools:
                revealed.extend(t.id for t in poke.tools if t)
            if poke.energyCards:
                revealed.extend(e.id for e in poke.energyCards if e)
                
    # Discard pile
    for card in ps.discard:
        if card is not None:
            revealed.append(card.id)
            
    return revealed


def create_dynamic_opponent_deck(revealed_ids: list[int]) -> list[int]:
    """Generates a realistic 60-card deck template using revealed cards & energy/evolution line inference."""
    deck = []
    card_db = {c.cardId: c for c in all_card_data()}
    
    # 1. Include revealed cards (up to 4 copies for plausible deck reconstruction)
    seen_counts = Counter(revealed_ids)
    for cid, count in seen_counts.items():
        deck.extend([cid] * min(4, max(count, 3)))
        
    # 2. Auto-infer evolution chain completion (e.g. Dreepy 119 -> Drakloak 120 -> Dragapult ex 121)
    revealed_mons = [card_db[cid] for cid in revealed_ids if cid in card_db and card_db[cid].cardType == CardType.POKEMON]
    for m in revealed_mons:
        for c in card_db.values():
            if getattr(c, "evolvesFrom", None) == m.name and c.cardId not in deck:
                deck.extend([c.cardId] * 2)

    # 3. Add High-Value Competitive Trainer Staples (Ultra Ball 1121, Switch 1123, Supporters)
    trainers = [1121, 1121, 1121, 1121, 1123, 1123, 1227, 1227, 1227, 1227, 1097, 1097, 1086, 1086, 1152, 1152, 1129, 1159]
    for t_id in trainers:
        if len(deck) >= 48:
            break
        deck.append(t_id)

    # 4. Fill Remaining with Energy Types up to exactly 60 Cards
    energy_pool = [2, 5, 1, 3]  # Fire, Psychic, Grass, Water
    idx = 0
    while len(deck) < 60:
        deck.append(energy_pool[idx % len(energy_pool)])
        idx += 1

    return deck[:60]


SIGNATURE_FINGERPRINTS = {
    # Cynthia's Line -> Garchomp ex
    379: "Rulebasedmodel_Garchomp_ex",      # Cynthia's Gible
    380: "Rulebasedmodel_Garchomp_ex",      # Cynthia's Gabite
    381: "Rulebasedmodel_Garchomp_ex",      # Cynthia's Garchomp ex
    341: "Rulebasedmodel_Garchomp_ex",      # Cynthia's Roselia
    1173: "Rulebasedmodel_Garchomp_ex",     # Cynthia's Power Weight
    
    # Ho-Oh / Fire Line
    357: "Rulebasedmodel_HoOh_HeartGold",   # Ethan's Ho-Oh ex
    46:  "Rulebasedmodel_HoOh_HeartGold",   # Gouging Fire ex
    1215: "Rulebasedmodel_HoOh_HeartGold",  # Ethan's Adventure
    
    # Mega Starmie Line
    1031: "Rulebasedmodel_Starmie_ex_2",    # Mega Starmie ex
    721: "Rulebasedmodel_Starmie_ex_2",     # Kyogre
    1145: "Rulebasedmodel_Starmie_ex_2",    # Mega Signal
    
    # Steven's Metagross Line
    641: "Rulebasedmodel_Metagross_Grass",  # Steven's Metagross ex
    639: "Rulebasedmodel_Metagross_Grass",  # Steven's Beldum
    640: "Rulebasedmodel_Metagross_Grass",  # Steven's Metang
    
    # Dragapult Line
    119: "Rulebasedmodel_Dragapult",        # Dreepy
    120: "Rulebasedmodel_Dragapult",        # Drakloak
    121: "Rulebasedmodel_Dragapult",        # Dragapult ex
    
    # Hydrapple / Ogerpon Line
    96:  "Rulebasedmodel_Hydrapple_Ogerpon", # Teal Mask Ogerpon ex
    150: "Rulebasedmodel_Hydrapple_Ogerpon", # Hydrapple ex
    117: "Rulebasedmodel_Garchomp_ex_2",    # Cornerstone Mask Ogerpon ex
    
    # Team Rocket Lines
    891: "Rulebasedmodel_Honchkrow",        # Team Rocket's Honchkrow
    463: "Rulebasedmodel_Honchkrow",        # Team Rocket's Murkrow
    473: "Rulebasedmodel_Honchkrow",        # Team Rocket's Porygon
    
    # Typhlosion Line
    352: "Rulebasedmodel_Typhlosion",       # Cyndaquil
    354: "Rulebasedmodel_Typhlosion",       # Typhlosion
    
    # Lopunny Line
    65:  "Rulebasedmodel_Lopunny",          # Buneary
    66:  "Rulebasedmodel_Lopunny",          # Lopunny
    
    # Grimmsnarl Line
    646: "Rulebasedmodel_Grimmsnarl_ex",    # Impidimp
    648: "Rulebasedmodel_Grimmsnarl_ex",    # Grimmsnarl ex
    
    # Alakazam Line
    265: "Rulebasedmodel_Iono",             # Abra
    271: "Rulebasedmodel_Iono",             # Alakazam ex
}


def sample_opponent_belief_deck(revealed_ids: list[int], opponent_decks: dict) -> list[int]:
    """Computes Bayesian belief probabilities over known archetypes + dynamic template, sampling a belief deck for MCTS."""
    if not revealed_ids:
        sample_key = random.choice(list(opponent_decks.keys()))
        return opponent_decks[sample_key]

    # Instant Signature Fingerprint Check (Option 3): Lock in 99% confidence on Turn 1!
    for cid in revealed_ids:
        if cid in SIGNATURE_FINGERPRINTS:
            matched_archetype = SIGNATURE_FINGERPRINTS[cid]
            if matched_archetype in opponent_decks:
                return opponent_decks[matched_archetype]

    scores = {}
    r_counts = Counter(revealed_ids)
    r_len = max(1, len(revealed_ids))

    for name, deck in opponent_decks.items():
        d_counts = Counter(deck)
        matches = sum(min(cnt, d_counts.get(cid, 0)) for cid, cnt in r_counts.items())
        scores[name] = matches / r_len

    max_score = max(scores.values()) if scores else 0.0
    
    # Low max_score means low confidence in known archetypes -> favors dynamic template
    dynamic_score = max(0.0, 1.0 - max_score * 1.5)
    
    # Softmax temperature-scaled belief probabilities
    temp = 4.0
    exp_scores = {k: math.exp(v * temp) for k, v in scores.items()}
    exp_scores["__DYNAMIC__"] = math.exp(dynamic_score * temp)
    
    total_prob = sum(exp_scores.values())
    keys = list(exp_scores.keys())
    weights = [exp_scores[k] / total_prob for k in keys]
    
    chosen_key = random.choices(keys, weights=weights, k=1)[0]
    
    if chosen_key == "__DYNAMIC__":
        return create_dynamic_opponent_deck(revealed_ids)
    else:
        return opponent_decks[chosen_key]


def get_own_visible_card_ids(obs, your_index: int) -> list[int]:
    """Scans all visible own zones (active, bench, discard, hand) to find card IDs."""
    visible = []
    ps = obs.current.players[your_index]
    
    # Hand
    for card in ps.hand:
        if card is not None:
            visible.append(card.id)
            
    # Active Pokémon + attached cards (energies, tools)
    for poke in ps.active:
        if poke is not None:
            visible.append(poke.id)
            if poke.tools:
                visible.extend(t.id for t in poke.tools if t)
            if poke.energyCards:
                visible.extend(e.id for e in poke.energyCards if e)
                
    # Bench Pokémon + attached cards
    for poke in ps.bench:
        if poke is not None:
            visible.append(poke.id)
            if poke.tools:
                visible.extend(t.id for t in poke.tools if t)
            if poke.energyCards:
                visible.extend(e.id for e in poke.energyCards if e)
                
    # Discard pile
    for card in ps.discard:
        if card is not None:
            visible.append(card.id)
            
    return visible


def mcts_agent(obs_dict: dict, your_deck: list[int], model: MyModel, search_count: int = None, temperature: float = 0.0, force_action: list[int] = None, opponent_name: str = "unknown", epoch: int = 1) -> tuple[list[int], LearnSample]:
    """Perform MCTS exploration and select the best action list, returning it and a training sample."""
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    
    # Dynamically sample opponent belief deck using Bayesian Information Set probabilities
    opp_index = 1 - your_index
    revealed_ids = get_opponent_revealed_card_ids(obs, opp_index)
    matched_deck = sample_opponent_belief_deck(revealed_ids, OPPONENT_DECKS)
    
    # Use Counter for O(1) per-card removal instead of O(n) list.remove()
    remaining_counter = Counter(matched_deck)
    for cid in revealed_ids:
        if remaining_counter.get(cid, 0) > 0:
            remaining_counter[cid] -= 1
    remaining_cards = []
    for cid, cnt in remaining_counter.items():
        remaining_cards.extend([cid] * cnt)
    random.shuffle(remaining_cards)
    
    deck_count = state.players[opp_index].deckCount
    prize_count = len(state.players[opp_index].prize)
    hand_count = state.players[opp_index].handCount
    
    total_needed = deck_count + prize_count + hand_count
    if len(remaining_cards) < total_needed:
        # Fallback padding if deck mismatch occurs
        remaining_cards.extend([3] * (total_needed - len(remaining_cards)))
        
    opp_deck_sampled = remaining_cards[:deck_count]
    opp_prize_sampled = remaining_cards[deck_count:deck_count + prize_count]
    opp_hand_sampled = remaining_cards[deck_count + prize_count:deck_count + prize_count + hand_count]
    
    # Sample own hidden zones correctly — Counter for O(1) per-card removal
    own_counter = Counter(your_deck)
    own_visible = get_own_visible_card_ids(obs, your_index)
    for cid in own_visible:
        if own_counter.get(cid, 0) > 0:
            own_counter[cid] -= 1
    own_remaining = []
    for cid, cnt in own_counter.items():
        own_remaining.extend([cid] * cnt)
    random.shuffle(own_remaining)
    
    own_prize_count = len(state.players[your_index].prize)
    own_deck_count = state.players[your_index].deckCount
    
    # Safe fallback if count mismatch
    total_own_needed = own_prize_count + own_deck_count
    if len(own_remaining) < total_own_needed:
        own_remaining.extend([3] * (total_own_needed - len(own_remaining)))
        
    your_prize_sampled = own_remaining[:own_prize_count]
    your_deck_sampled = own_remaining[own_prize_count:own_prize_count + own_deck_count]
    
    search_state = search_begin(
        obs,
        your_deck=your_deck_sampled,
        your_prize=your_prize_sampled,
        opponent_deck=opp_deck_sampled,
        opponent_prize=opp_prize_sampled,
        opponent_hand=opp_hand_sampled,
        opponent_active=[1072] if len(active) > 0 and active[0] is None else []
    )
    
    root, sample = create_node(None, search_state, your_index, your_deck, model, epoch=epoch)

    # Add Dirichlet noise to root prior for exploration (AlphaZero-style)
    # Without noise, MCTS always explores the same paths from the NN prior,
    # causing mode collapse (the agent repeatedly selects "end turn").
    if len(root.children) > 0:
        dir_alpha = 0.3  # TCG has moderate action space
        turn = state.turn if (state is not None) else 0
        noise_frac = 0.25  # AlphaZero standard; 0.40 was over-randomising with only 50 sims
        noise = [random.gammavariate(dir_alpha, 1.0) for _ in root.children]
        noise_sum = sum(noise) + 1e-8
        noise = [n / noise_sum for n in noise]
        for i, child in enumerate(root.children):
            child.prob = (1.0 - noise_frac) * child.prob + noise_frac * noise[i]

    # Dynamic Simulation Count based on game urgency (Option 2: 350 MCTS Rollout Expansion)
    # Opening (Turns 1-3): 120 sims (fast opening setup)
    # Midgame (Turns 4-9): 200 sims (balanced board development)
    # Critical Endgame / Clutch Turns (Turn 10+ or <=3 prizes remaining or Active HP < 100): 350 sims (Maximum 2x thinking depth)
    if search_count is None or search_count >= 200:
        turn = state.turn if (state is not None) else 1
        my_prizes = len(state.players[your_index].prize) if (state and len(state.players) > your_index) else 6
        opp_prizes = len(state.players[opp_index].prize) if (state and len(state.players) > opp_index) else 6
        min_prizes = min(my_prizes, opp_prizes)
        
        my_active_pk = state.players[your_index].active[0] if (state and len(state.players[your_index].active) > 0 and state.players[your_index].active[0]) else None
        active_hp = my_active_pk.hp if my_active_pk else 300
        
        is_clutch_turn = (min_prizes <= 3) or (active_hp < 100) or (turn >= 10)
        
        if is_clutch_turn:
            dynamic_search_count = 350  # 350 MCTS rollouts for clutch game-deciding turns
        elif turn <= 3:
            dynamic_search_count = 120  # Fast setup
        else:
            dynamic_search_count = 200  # Balanced midgame
    else:
        dynamic_search_count = search_count

    # Caps simulations in the Kaggle runtime environment if total time is constrained
    IS_KAGGLE = os.path.exists('/kaggle_simulations/agent') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    if IS_KAGGLE and search_count is None:
        dynamic_search_count = min(dynamic_search_count, 250)

    # Search loop
    for _ in range(dynamic_search_count):
        current = root
        while True:
            value = -1e9
            # PUCT: Q(s,a) + C_PUCT * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))
            # C_PUCT is a fixed constant per AlphaZero (not scaled by parent visits).
            puct_scale = C_PUCT * math.sqrt(max(1, current.visit))
            next_child = None
            for child in current.children:
                visit = 0
                if child.node is None:
                    v = current.total / max(1, current.visit)
                else:
                    v = child.node.total / max(1, child.node.visit)
                    visit = child.node.visit
                
                if current.state.observation.current.yourIndex != your_index:
                    v = -v
                exp_bonus = 0.0
                if child.select_option is not None and hasattr(current.state, "observation"):
                    exp_bonus, _ = get_expert_bonus(current.state.observation, child.select_option, opponent_name=opponent_name)
                v += puct_scale * child.prob / (1 + visit) + EXPERT_WEIGHT * exp_bonus
                if value < v:
                    value = v
                    next_child = child
            
            if next_child.node is None:
                search_state = search_step(current.state.searchId, next_child.select)
                next_child.node, _ = create_node(current, search_state, your_index, your_deck, model, epoch=epoch)
                next_child.node.backprop(next_child.node.value)
                break
            else:
                current = next_child.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

    # Select action according to temperature:
    #   temperature=0.0 → argmax on visit counts (deterministic / evaluation mode)
    #   temperature=1.0 → sample proportional to visit counts (AlphaZero training mode)
    visited_children = [c for c in root.children if c.node is not None]
    
    if temperature > 0.0 and visited_children:
        # AlphaZero-style: sample proportional to N(s,a)^(1/τ)
        visits = [c.node.visit ** (1.0 / temperature) for c in visited_children]
        total_v = sum(visits)
        if total_v > 0:
            weights = [v / total_v for v in visits]
            max_child = random.choices(visited_children, weights=weights, k=1)[0]
        else:
            max_child = visited_children[0]
    else:
        # Argmax (evaluation / greedy mode)
        max_child = None
        max_visit = -1
        for child in visited_children:
            if child.node.visit > max_visit:
                max_child = child
                max_visit = child.node.visit

    # Fallback: if no children were visited (e.g. search_count=0), pick highest-prior child
    if max_child is None and root.children:
        max_child = max(root.children, key=lambda c: c.prob)

    # Handle terminal root state — sample is None when the game is already over at root
    if sample is None:
        search_end()
        sel = max_child.select if max_child is not None else []
        return (sel, LearnSample(0.0, [], SparseVector(), SparseVector()))

    # Generate targets/labels for training
    # Value target: root mean value
    sample.value = root.total / root.visit

    # Policy target: visit count distribution (AlphaZero-style)
    # Visit counts are more robust than Q-value differences with limited simulations.
    total_child_visits = sum(
        child.node.visit for child in root.children if child.node is not None
    )
    if force_action is not None:
        found = False
        for i in range(len(root.children)):
            child = root.children[i]
            if child.select == force_action:
                sample.policy[i] = 1.0
                found = True
            else:
                sample.policy[i] = 0.0
        if not found and len(root.children) > 0:
            sample.policy[0] = 1.0
    elif total_child_visits > 0:
        for i in range(len(root.children)):
            child = root.children[i]
            if child.node is not None:
                sample.policy[i] = child.node.visit / total_child_visits
            else:
                sample.policy[i] = 0.0
    else:
        # Fallback: uniform if no visits
        n = len(root.children)
        for i in range(n):
            sample.policy[i] = 1.0 / n

    search_end()
    final_selected = force_action if force_action is not None else (max_child.select if max_child is not None else [])
    return (final_selected, sample)


def random_agent(obs_dict: dict) -> list[int]:
    """Random baseline agent."""
    obs = to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)


# Cached global model and deck to avoid repeated loading
_model = None
_deck = None


def agent(obs_dict: dict) -> list[int]:
    """Kaggle Competition Entry point function."""
    global _model, _deck
    
    # Load deck list
    if _deck is None:
        if "__file__" in globals():
            base_path = os.path.dirname(os.path.abspath(__file__))
        else:
            base_path = os.getcwd()
        deck_path = os.path.join(base_path, "deck.csv")
        try:
            with open(deck_path, "r") as f:
                _deck = [int(line.strip()) for line in f if line.strip()]
        except Exception:
            # Fallback to exact Team Rocket 60-card decklist
            _deck = [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 15, 15, 15, 15, 400, 400, 400, 401, 401, 401, 414, 414, 431, 431, 432, 434, 1094, 1094, 1097, 1097, 1116, 1121, 1121, 1121, 1121, 1129, 1134, 1134, 1134, 1134, 1152, 1159, 1175, 1216, 1216, 1216, 1216, 1217, 1217, 1218, 1218, 1218, 1219, 1220, 1220, 1220, 1227, 1227, 1257, 1257]

    # Check if this is the initial deck-registration step (obs.select is None)
    if obs_dict.get("select") is None:
        return _deck

    # Load model weights
    if _model is None:
        _model = MyModel(
            MODEL_D_MODEL,
            MODEL_NUM_HEADS,
            MODEL_D_FEEDFORWARD,
            MODEL_NUM_LAYERS_ENCODER,
            MODEL_NUM_LAYERS_DECODER
        )
        if "__file__" in globals():
            base_path = os.path.dirname(os.path.abspath(__file__))
        else:
            base_path = os.getcwd()
        candidate_paths = [
            os.path.join(base_path, "best_model.pth"),
            os.path.join(base_path, "model.pth"),
            os.path.join(os.getcwd(), "best_model.pth"),
            os.path.join(os.getcwd(), "model.pth"),
            "/kaggle_simulations/agent/best_model.pth",
            "/kaggle_simulations/agent/model.pth"
        ]
        
        loaded = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_path in candidate_paths:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        _model.load_state_dict(checkpoint["state_dict"])
                    else:
                        _model.load_state_dict(checkpoint)
                    print(f"Loaded RL model checkpoint from: {model_path}", file=sys.stderr)
                    loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load checkpoint {model_path}: {e}", file=sys.stderr)
        
        if not loaded:
            print("Warning: No checkpoint loaded! Operating on fresh RL model weights.", file=sys.stderr)
        
        _model = _model.to(device)
        _model.eval()

    # Determine adaptive search count budget based on current turn & remaining overage time
    obs = to_observation_class(obs_dict)
    turn = obs.current.turn if (obs.current is not None) else 0
    remaining_time = float(obs_dict.get('remainingOverageTime', 600.0))
    
    # Check if active is walled (Mewtwo ex active against opponent Mimikyu)
    active_is_walled = False
    try:
        your_index = obs.current.yourIndex
        my_active = obs.current.players[your_index].active
        opp_active = obs.current.players[1 - your_index].active
        if my_active and opp_active:
            if my_active[0].id == 431 and opp_active[0].id == 434:
                active_is_walled = True
    except Exception:
        pass

    # Check if context is MAIN (critical turn-beginning choice)
    is_main_context = False
    try:
        if obs.select is not None and obs.select.context == SelectContext.MAIN:
            is_main_context = True
    except Exception:
        pass

    IS_KAGGLE = os.path.exists('/kaggle_simulations/agent') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    if remaining_time < 60.0:
        # Emergency fast play mode to prevent TIMEOUT when remaining time is low (<60s)
        search_count = 5
    elif remaining_time < 150.0:
        # Low time budget mode (<150s)
        search_count = 8
    elif remaining_time < 300.0:
        # Moderate time budget mode (<300s)
        if active_is_walled or is_main_context:
            search_count = 20 if IS_KAGGLE else 50
        else:
            search_count = 10 if IS_KAGGLE else 25
    elif IS_KAGGLE:
        # Standard Kaggle budget with comfortable time remaining (>300s)
        if active_is_walled or is_main_context:
            search_count = 35  # Boost budget for critical decisions on Kaggle
        elif turn <= 3:
            search_count = 20
        elif turn <= 8:
            search_count = 15
        else:
            search_count = 10
    else:
        # Standard local/eval budget
        if active_is_walled or is_main_context:
            search_count = 180  # Boost budget to 150-200 for critical/walled decisions
        elif turn <= 3:
            search_count = 150
        elif turn <= 8:
            search_count = 100
        else:
            search_count = 50

    try:
        with torch.inference_mode():
            action, _ = mcts_agent(obs_dict, _deck, _model, search_count=search_count)
        return action
    except Exception as e:
        print(f"MCTS Agent crashed: {e}. Falling back to default action.", file=sys.stderr)
        obs = to_observation_class(obs_dict)
        if obs.select and obs.select.option:
            return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)
        return []
