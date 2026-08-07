import glob
import math
import os
import random
import sys
import torch
from collections import Counter

"""
Pure AlphaZero Inference & MCTS Strategy Engine for Pokémon TCG AI Agent
=========================================================================
Architecture: Pure AlphaZero (DeepMind Architecture)
  1. Game State Observation -> Encoded via Transformer State Encoder
  2. Neural Network Evaluation -> (Value V(s) in [-1, +1], Policy Logits P(s, a))
  3. MCTS PUCT Search -> Guided by Neural Network Policy & Value Predictions
  4. Final Action Selection -> Visit Count Distribution N(s, a)^(1/tau)

Decoupling Principle:
  - Inference-time decisions in agent.py are driven by best_model.pth & MCTS.
  - Hardcoded move ordering and overpowering logit offsets are removed.
  - Legality constraints & Kaggle interfaces are 100% preserved.
"""

try:
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
except ImportError:
    from src.model import (
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
    from expert_knowledge import get_expert_bonus, EXPERT_WEIGHT, USE_EXPERT_GUIDANCE, expert_scale
except ImportError:
    from src.expert_knowledge import get_expert_bonus, EXPERT_WEIGHT, USE_EXPERT_GUIDANCE, expert_scale

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
evolves_from_map: dict[str, list] = {}
for _c in card_table.values():
    _ef = getattr(_c, "evolvesFrom", None)
    if _ef:
        evolves_from_map.setdefault(_ef, []).append(_c)

SEARCH_COUNT = 200  # MCTS Search count — ≥200 needed for meaningful visit differentiation
C_PUCT = 1.25       # AlphaZero PUCT exploration constant (fixed, per DeepMind AlphaZero)


class LearnSample:
    """Single Training Sample collected during gameplay/self-play."""
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec: SparseVector):
        self.value = value        # Expected win outcome in [-1, +1]
        self.policy = policy      # Action probability distribution
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
    value: float          # Self value from Neural Network
    total: float          # Total accumulated value from backpropagation
    visit: int            # Visit count N(s)
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


def _extract_obs_info(obs_input):
    """Safely extracts (obs_obj, current_state, result, your_index) from dict, SearchState, or Observation."""
    if hasattr(obs_input, "observation"):
        obs_obj = obs_input.observation
    else:
        obs_obj = obs_input

    if hasattr(obs_obj, "current"):
        current_state = obs_obj.current
    elif isinstance(obs_input, dict) and "current" in obs_input:
        current_state = obs_input["current"]
    else:
        current_state = obs_obj

    result = getattr(current_state, "result", -1)
    if isinstance(current_state, dict):
        result = current_state.get("result", -1)

    your_index = getattr(current_state, "yourIndex", 0)
    if isinstance(current_state, dict):
        your_index = current_state.get("yourIndex", 0)

    return obs_obj, current_state, result, your_index


def detect_opponent_archetype(obs, your_index: int) -> str:
    """Detect opponent deck archetype from revealed active, bench, and discard cards."""
    obs_obj, current_state, _, _ = _extract_obs_info(obs)
    if current_state is None:
        return "GENERIC"
    try:
        opp_index = 1 - your_index
        players = getattr(current_state, "players", [])
        if isinstance(current_state, dict):
            players = current_state.get("players", [])

        if len(players) > opp_index:
            opp_player = players[opp_index]
            visible_cards = []
            active = getattr(opp_player, "active", []) if not isinstance(opp_player, dict) else opp_player.get("active", [])
            bench = getattr(opp_player, "bench", []) if not isinstance(opp_player, dict) else opp_player.get("bench", [])
            discard = getattr(opp_player, "discard", []) if not isinstance(opp_player, dict) else opp_player.get("discard", [])

            for card in list(active) + list(bench) + list(discard):
                if card:
                    name = getattr(card, 'name', '').lower() if not isinstance(card, dict) else card.get('name', '').lower()
                    if name:
                        visible_cards.append(name)
            card_str = " ".join(visible_cards)
            if any(k in card_str for k in ("dreepy", "drakloak", "dragapult", "froslass", "dusknoir", "dusclops", "munkidori")):
                return "DRAGAPULT"
            elif any(k in card_str for k in ("snover", "abomasnow")):
                return "ABOMASNOW"
            elif any(k in card_str for k in ("grimmsnarl", "grim", "impidimp")):
                return "GRIMMSNARL"
            elif any(k in card_str for k in ("charizard", "zard", "charmander")):
                return "CHARIZARD"
            elif any(k in card_str for k in ("regidrago", "giratina", "roaring", "dragon", "koraidon", "miraidon")):
                return "DRAGON"
            elif any(k in card_str for k in ("iono", "pidgeot", "snorlax")):
                return "IONO_CONTROL"
            elif card_str:
                return card_str
    except Exception:
        pass
    return "GENERIC"


def create_node(parent: Node | None,
                search_state: SearchState,
                your_index: int,
                your_deck: list[int],
                model: MyModel,
                epoch: int = 1
) -> tuple[Node, LearnSample | None]:
    """
    Creates a new MCTS node and evaluates state & candidate actions using Neural Network.
    Follows Pure AlphaZero architecture:
      - Valid options enumerated cleanly.
      - Neural Network policy logits P(s, a) and value V(s) drive exploration.
      - Optional soft expert priors decay smoothly over training epochs.
    """
    node = Node(parent, search_state)
    obs_obj, state, result, your_idx = _extract_obs_info(search_state)
    if your_index is None:
        your_index = your_idx

    if result >= 0:
        # Battle finished: Terminal Node Valuation
        if result == 2:
            node.value = 0.0
        elif result == your_index:
            node.value = 1.0
        else:
            node.value = -1.0
        sample = None
    else:
        # Layer 1: Legality Masking & Action Combination Enumeration
        actions = []
        select_obj = getattr(obs_obj, "select", None)
        options = getattr(select_obj, "option", []) if select_obj else []
        archetype = detect_opponent_archetype(obs_obj, your_index)

        # Enumerate valid combinations respecting minCount <= len(selected) <= maxCount
        n = len(options)
        min_k = getattr(select_obj, "minCount", 1) if select_obj else 1
        max_k = getattr(select_obj, "maxCount", 1) if select_obj else 1
        if isinstance(obs_obj, dict):
            sel_dict = obs_obj.get("select", {})
            min_k = sel_dict.get("minCount", min_k)
            max_k = sel_dict.get("maxCount", max_k)

        max_k_valid = min(max_k, n)
        min_k_valid = max(0, min(min_k, max_k_valid))

        for k_val in range(min_k_valid, max_k_valid + 1):
            if k_val == 0:
                actions.append([])
                if len(actions) >= 256:
                    break
                continue

            comb_positions = list(range(k_val))
            while True:
                actions.append(list(comb_positions))
                if len(actions) >= 256:
                    break
                
                # Standard next combination algorithm of size k_val over options [0, n-1]
                for i in range(k_val):
                    idx = k_val - i - 1
                    if comb_positions[idx] < n - i - 1:
                        comb_positions[idx] += 1
                        for j in range(idx + 1, k_val):
                            comb_positions[j] = comb_positions[j - 1] + 1
                        break
                else:
                    break
            if len(actions) >= 256:
                break

        if not actions:
            if min_k == 0:
                actions = [[]]
            elif n > 0:
                actions = [[i] for i in range(min(n, max(1, min_k)))]
            else:
                actions = [[0]]


        # Layer 3: Neural Network Evaluation (State Encoder + Action Decoder)
        sv_enc = get_encoder_input(obs_obj, your_deck)
        sv_dec = get_decoder_input(obs_obj, actions)
        value, policy = eval_nn(sv_enc, sv_dec, model)

        v = value
        curr_index = getattr(state, "yourIndex", 0) if not isinstance(state, dict) else state.get("yourIndex", 0)
        if curr_index != your_index:
            v = -v
        node.value = v

        # Layer 2: Soft Prior Integration (Probability-Space Blending: NN Logits + Soft Expert Curriculum Prior)
        n_actions = len(actions)
        nn_logits = list(policy[:n_actions]) if n_actions > 0 else []

        # 1. Compute raw NN prior probabilities via Softmax over neural logits
        if nn_logits:
            max_nn_logit = max(nn_logits)
            exp_nn = [math.exp(l - max_nn_logit) for l in nn_logits]
            sum_exp_nn = sum(exp_nn)
            nn_probs = [e / sum_exp_nn for e in exp_nn] if sum_exp_nn > 0 else [1.0 / n_actions] * n_actions
        else:
            nn_probs = []

        curriculum_scale = expert_scale(epoch) if USE_EXPERT_GUIDANCE else 0.0

        if curriculum_scale > 0.0 and nn_probs:
            has_constructive = False
            for opt in options:
                opt_type = getattr(opt, "type", -1) if not isinstance(opt, dict) else opt.get("type", -1)
                if opt_type in (OptionType.ATTACK, OptionType.ATTACH, OptionType.EVOLVE, OptionType.PLAY, OptionType.ABILITY):
                    has_constructive = True
                    break

            expert_scores = [0.0] * n_actions
            for i in range(n_actions):
                bias = 0.0
                has_end = False

                for opt_idx in actions[i]:
                    if opt_idx < len(options):
                        opt = options[opt_idx]
                        opt_type = getattr(opt, "type", -1) if not isinstance(opt, dict) else opt.get("type", -1)
                        if opt_type == OptionType.END:
                            has_end = True

                        try:
                            exp_bonus, _ = get_expert_bonus(obs_obj, opt, opponent_name=archetype, epoch=epoch)
                            if exp_bonus <= -0.20:
                                bias -= 1.0  # Soft prior guidance (non-prohibitive, preserves AlphaZero exploration)
                            else:
                                bias += exp_bonus * 1.0
                        except Exception:
                            pass

                if has_end and has_constructive:
                    bias -= 1.0

                expert_scores[i] = bias

            max_exp_score = max(expert_scores) if expert_scores else 0.0
            exp_e = [math.exp(s - max_exp_score) for s in expert_scores]
            sum_exp_e = sum(exp_e)
            expert_probs = [e / sum_exp_e for e in exp_e] if sum_exp_e > 0 else [1.0 / n_actions] * n_actions

            # Soft Probability Blend (Max 15% expert prior weight, 85% neural network)
            alpha_blend = 0.15 * curriculum_scale
            final_priors = [(1.0 - alpha_blend) * p_nn + alpha_blend * p_exp for p_nn, p_exp in zip(nn_probs, expert_probs)]
        else:
            final_priors = nn_probs

        for i in range(n_actions):
            first_opt = options[actions[i][0]] if (options and actions[i] and actions[i][0] < len(options)) else None
            node.children.append(Child(actions[i], final_priors[i] if i < len(final_priors) else 1.0 / n_actions, select_option=first_opt))

        target_policy_probs = [c.prob for c in node.children]
        sample = LearnSample(value, target_policy_probs, sv_enc, sv_dec)



    return (node, sample)


# --- Opponent Deck Database for Self-Play & Matchup Simulation ---
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
    'ek': [1158, 721, 721, 722, 722, 722, 723, 723, 723, 723, 1145, 1145, 1145, 1145, 1205, 1205, 1227, 1227, 1227, 1227, 1235, 1235, 1235, 1235, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'Gholdengo Lunatone': [186, 186, 186, 186, 191, 676, 676, 675, 675, 311, 547, 140, 695, 1182, 1182, 1182, 1184, 1142, 1142, 1142, 1086, 1088, 1174, 1174, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 700, 700, 700, 1213, 1213, 1213, 1213, 1123, 1123, 1118, 1118, 1118, 1118, 1121, 1121, 1121, 1121, 1119, 1119, 1097, 1177, 1086, 1086],
    'Hermes Lu': [119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 140, 184, 235, 235, 1071, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1120, 1120, 1120, 1120, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1156, 1182, 1182, 1182, 1198, 1198, 1198, 1198, 1210, 1210, 1227, 1227, 1227, 1227, 1256, 1256, 2, 2, 2, 2, 5, 5, 5, 5],
}


def rule_based_opponent_agent(deck_name: str, obs: dict) -> list[int]:
    """Fallback rule-based opponent baseline for training self-play warmups."""
    options = obs.get("select", {}).get("option", []) if isinstance(obs, dict) else getattr(obs.select, "option", [])
    if not options:
        return [0]

    for idx, opt in enumerate(options):
        opt_type = opt.get("type", -1) if isinstance(opt, dict) else getattr(opt, "type", -1)
        if opt_type == OptionType.ATTACK:
            return [idx]
    for idx, opt in enumerate(options):
        opt_type = opt.get("type", -1) if isinstance(opt, dict) else getattr(opt, "type", -1)
        if opt_type == OptionType.EVOLVE:
            return [idx]
    for idx, opt in enumerate(options):
        opt_type = opt.get("type", -1) if isinstance(opt, dict) else getattr(opt, "type", -1)
        if opt_type == OptionType.ATTACH:
            return [idx]
    for idx, opt in enumerate(options):
        opt_type = opt.get("type", -1) if isinstance(opt, dict) else getattr(opt, "type", -1)
        if opt_type == OptionType.PLAY:
            return [idx]

    return [0]


def random_agent(obs: dict) -> list[int]:
    """Random baseline action selector respecting minCount and maxCount bounds."""
    options = obs.get("select", {}).get("option", []) if isinstance(obs, dict) else getattr(getattr(obs, "select", None), "option", [])
    min_k = obs.get("select", {}).get("minCount", 1) if isinstance(obs, dict) else getattr(getattr(obs, "select", None), "minCount", 1)
    max_k = obs.get("select", {}).get("maxCount", 1) if isinstance(obs, dict) else getattr(getattr(obs, "select", None), "maxCount", 1)

    n = len(options)
    if n == 0:
        return [0]

    k_val = min_k if min_k <= max_k else max_k
    k_val = max(0, min(k_val, n))

    if k_val == 0:
        return []

    return list(range(k_val))



def get_opponent_revealed_card_ids(obs, opponent_index: int) -> list[int]:
    """Scans visible zones (active, bench, discard) to find cards played by the opponent."""
    revealed = []
    obs_obj, state, _, _ = _extract_obs_info(obs)
    if state is None or not hasattr(state, "players"):
        return revealed
    ps = state.players[opponent_index]
    
    # Active Pokémon + attached cards
    active = getattr(ps, "active", []) if not isinstance(ps, dict) else ps.get("active", [])
    for poke in list(active):
        if poke is not None:
            pid = getattr(poke, 'id', None) if not isinstance(poke, dict) else poke.get('id')
            if pid:
                revealed.append(pid)
            tools = getattr(poke, 'tools', []) if not isinstance(poke, dict) else poke.get('tools', [])
            for t in list(tools):
                if t:
                    tid = getattr(t, 'id', None) if not isinstance(t, dict) else t.get('id')
                    if tid:
                        revealed.append(tid)
            energies = getattr(poke, 'energyCards', []) if not isinstance(poke, dict) else poke.get('energyCards', [])
            for e in list(energies):
                if e:
                    eid = getattr(e, 'id', None) if not isinstance(e, dict) else e.get('id')
                    if eid:
                        revealed.append(eid)
                
    # Bench Pokémon + attached cards
    bench = getattr(ps, "bench", []) if not isinstance(ps, dict) else ps.get("bench", [])
    for poke in list(bench):
        if poke is not None:
            pid = getattr(poke, 'id', None) if not isinstance(poke, dict) else poke.get('id')
            if pid:
                revealed.append(pid)
            tools = getattr(poke, 'tools', []) if not isinstance(poke, dict) else poke.get('tools', [])
            for t in list(tools):
                if t:
                    tid = getattr(t, 'id', None) if not isinstance(t, dict) else t.get('id')
                    if tid:
                        revealed.append(tid)
            energies = getattr(poke, 'energyCards', []) if not isinstance(poke, dict) else poke.get('energyCards', [])
            for e in list(energies):
                if e:
                    eid = getattr(e, 'id', None) if not isinstance(e, dict) else e.get('id')
                    if eid:
                        revealed.append(eid)
                
    # Discard pile
    discard = getattr(ps, "discard", []) if not isinstance(ps, dict) else ps.get("discard", [])
    for card in list(discard):
        if card is not None:
            cid = getattr(card, 'id', None) if not isinstance(card, dict) else card.get('id')
            if cid:
                revealed.append(cid)
            
    return revealed


def create_dynamic_opponent_deck(revealed_ids: list[int]) -> list[int]:
    """Generates a realistic 60-card deck template using revealed cards & energy/evolution line inference."""
    deck = []
    seen_counts = Counter(revealed_ids)
    for cid, count in seen_counts.items():
        deck.extend([cid] * min(4, max(count, 3)))
        
    revealed_mons = [card_table[cid] for cid in revealed_ids if cid in card_table and card_table[cid].cardType == CardType.POKEMON]
    for m in revealed_mons:
        evos = evolves_from_map.get(m.name, [])
        for c in evos:
            if c.cardId not in deck:
                deck.extend([c.cardId] * 2)

    trainers = [1121, 1121, 1121, 1121, 1123, 1123, 1227, 1227, 1227, 1227, 1097, 1097, 1086, 1086, 1152, 1152, 1129, 1159]
    for t_id in trainers:
        if len(deck) >= 48:
            break
        deck.append(t_id)

    energy_pool = [2, 5, 1, 3]
    idx = 0
    while len(deck) < 60:
        deck.append(energy_pool[idx % len(energy_pool)])
        idx += 1

    return deck[:60]


SIGNATURE_FINGERPRINTS = {
    379: "Rulebasedmodel_Garchomp_ex",
    380: "Rulebasedmodel_Garchomp_ex",
    381: "Rulebasedmodel_Garchomp_ex",
    341: "Rulebasedmodel_Garchomp_ex",
    1173: "Rulebasedmodel_Garchomp_ex",
    357: "Rulebasedmodel_HoOh_HeartGold",
    46:  "Rulebasedmodel_HoOh_HeartGold",
    1215: "Rulebasedmodel_HoOh_HeartGold",
    1031: "Rulebasedmodel_Starmie_ex_2",
    721: "Rulebasedmodel_Starmie_ex_2",
    1145: "Rulebasedmodel_Starmie_ex_2",
    641: "Rulebasedmodel_Metagross_Grass",
    639: "Rulebasedmodel_Metagross_Grass",
    640: "Rulebasedmodel_Metagross_Grass",
    119: "Rulebasedmodel_Dragapult",
    120: "Rulebasedmodel_Dragapult",
    121: "Rulebasedmodel_Dragapult ex",
    96:  "Rulebasedmodel_Hydrapple_Ogerpon",
    150: "Rulebasedmodel_Hydrapple_Ogerpon",
    117: "Rulebasedmodel_Garchomp_ex_2",
    891: "Rulebasedmodel_Honchkrow",
    463: "Rulebasedmodel_Honchkrow",
    473: "Rulebasedmodel_Honchkrow",
    352: "Rulebasedmodel_Typhlosion",
    354: "Rulebasedmodel_Typhlosion",
    65:  "Rulebasedmodel_Lopunny",
    66:  "Rulebasedmodel_Lopunny",
    646: "Rulebasedmodel_Grimmsnarl_ex",
    648: "Rulebasedmodel_Grimmsnarl_ex",
    265: "Rulebasedmodel_Iono",
    271: "Rulebasedmodel_Iono",
}


def sample_opponent_belief_deck(revealed_ids: list[int], opponent_decks: dict) -> list[int]:
    """Computes Bayesian belief probabilities over known archetypes + dynamic template, sampling a belief deck for MCTS."""
    if not revealed_ids:
        sample_key = random.choice(list(opponent_decks.keys()))
        return opponent_decks[sample_key]

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
    dynamic_score = max(0.0, 1.0 - max_score * 1.5)
    
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
    obs_obj, state, _, _ = _extract_obs_info(obs)
    if state is None or not hasattr(state, "players"):
        return visible
    ps = state.players[your_index]
    
    # Hand
    hand = getattr(ps, "hand", []) if not isinstance(ps, dict) else ps.get("hand", [])
    for card in list(hand):
        if card is not None:
            cid = getattr(card, 'id', None) if not isinstance(card, dict) else card.get('id')
            if cid:
                visible.append(cid)
            
    # Active Pokémon + attached cards
    active = getattr(ps, "active", []) if not isinstance(ps, dict) else ps.get("active", [])
    for poke in list(active):
        if poke is not None:
            pid = getattr(poke, 'id', None) if not isinstance(poke, dict) else poke.get('id')
            if pid:
                visible.append(pid)
            tools = getattr(poke, 'tools', []) if not isinstance(poke, dict) else poke.get('tools', [])
            for t in list(tools):
                if t:
                    tid = getattr(t, 'id', None) if not isinstance(t, dict) else t.get('id')
                    if tid:
                        visible.append(tid)
            energies = getattr(poke, 'energyCards', []) if not isinstance(poke, dict) else poke.get('energyCards', [])
            for e in list(energies):
                if e:
                    eid = getattr(e, 'id', None) if not isinstance(e, dict) else e.get('id')
                    if eid:
                        visible.append(eid)
                
    # Bench Pokémon + attached cards
    bench = getattr(ps, "bench", []) if not isinstance(ps, dict) else ps.get("bench", [])
    for poke in list(bench):
        if poke is not None:
            pid = getattr(poke, 'id', None) if not isinstance(poke, dict) else poke.get('id')
            if pid:
                visible.append(pid)
            tools = getattr(poke, 'tools', []) if not isinstance(poke, dict) else poke.get('tools', [])
            for t in list(tools):
                if t:
                    tid = getattr(t, 'id', None) if not isinstance(t, dict) else t.get('id')
                    if tid:
                        visible.append(tid)
            energies = getattr(poke, 'energyCards', []) if not isinstance(poke, dict) else poke.get('energyCards', [])
            for e in list(energies):
                if e:
                    eid = getattr(e, 'id', None) if not isinstance(e, dict) else e.get('id')
                    if eid:
                        visible.append(eid)
                
    # Discard pile
    discard = getattr(ps, "discard", []) if not isinstance(ps, dict) else ps.get("discard", [])
    for card in list(discard):
        if card is not None:
            cid = getattr(card, 'id', None) if not isinstance(card, dict) else card.get('id')
            if cid:
                revealed.append(cid)
            
    return visible


def mcts_agent(obs: dict,
               your_deck: list[int],
               client: GPUInferenceClient | None = None,
               search_count: int = SEARCH_COUNT,
               temperature: float = 1.0,
               force_action: list[int] | None = None,
               opponent_name: str = "",
               epoch: int = 1
) -> tuple[list[int], LearnSample | None]:
    """
    AlphaZero MCTS Agent Implementation
    Executes MCTS tree search over search_count iterations, guided by Neural Network V(s) & P(s, a).
    Returns: (selected_action_indices, training_sample)
    """
    if force_action is not None:
        return force_action, None

    obs_dataclass = to_observation_class(obs)
    obs_obj, state, result, your_idx = _extract_obs_info(obs_dataclass)
    if result >= 0:
        return [0], None

    select_obj = getattr(obs_dataclass, "select", None)
    opts = getattr(select_obj, "option", []) if select_obj else []
    if len(opts) == 0:
        return [0], None
    if len(opts) == 1:
        try:
            sv_enc = get_encoder_input(obs_obj, your_deck)
            sv_dec = get_decoder_input(obs_obj, [[0]])
            single_sample = LearnSample(0.0, [1.0], sv_enc, sv_dec)
            single_sample.pred_val = 0.0
            return [0], single_sample
        except Exception:
            return [0], None

    search_state = None
    if getattr(obs_dataclass, "search_begin_input", None) is not None:
        try:
            opp_idx = 1 - your_idx
            revealed_ids = get_opponent_revealed_card_ids(obs_dataclass, opp_idx)
            matched_deck = sample_opponent_belief_deck(revealed_ids, OPPONENT_DECKS)

            remaining_counter = Counter(matched_deck)
            for cid in revealed_ids:
                if remaining_counter.get(cid, 0) > 0:
                    remaining_counter[cid] -= 1
            remaining_cards = []
            for cid, cnt in remaining_counter.items():
                remaining_cards.extend([cid] * cnt)
            random.shuffle(remaining_cards)

            opp_player = state.players[opp_idx] if state and hasattr(state, "players") else None
            deck_count = getattr(opp_player, "deckCount", 0) if opp_player else 0
            prize_count = len(getattr(opp_player, "prize", [])) if opp_player else 0
            hand_count = getattr(opp_player, "handCount", 0) if opp_player else 0

            total_needed = deck_count + prize_count + hand_count
            if len(remaining_cards) < total_needed:
                remaining_cards.extend([3] * (total_needed - len(remaining_cards)))

            opp_deck_sampled = remaining_cards[:deck_count]
            opp_prize_sampled = remaining_cards[deck_count:deck_count + prize_count]
            opp_hand_sampled = remaining_cards[deck_count + prize_count:deck_count + prize_count + hand_count]

            own_counter = Counter(your_deck)
            own_visible = get_own_visible_card_ids(obs_dataclass, your_idx)
            for cid in own_visible:
                if own_counter.get(cid, 0) > 0:
                    own_counter[cid] -= 1
            own_remaining = []
            for cid, cnt in own_counter.items():
                own_remaining.extend([cid] * cnt)
            random.shuffle(own_remaining)

            my_player = state.players[your_idx] if state and hasattr(state, "players") else None
            own_prize_count = len(getattr(my_player, "prize", [])) if my_player else 0
            own_deck_count = getattr(my_player, "deckCount", 0) if my_player else 0

            total_own_needed = own_prize_count + own_deck_count
            if len(own_remaining) < total_own_needed:
                own_remaining.extend([3] * (total_own_needed - len(own_remaining)))

            your_prize_sampled = own_remaining[:own_prize_count]
            your_deck_sampled = own_remaining[own_prize_count:own_prize_count + own_deck_count]
            active = getattr(opp_player, "active", []) if opp_player else []
            opp_active_sampled = [1072] if len(active) > 0 and active[0] is None else []

            search_state = search_begin(
                obs_dataclass,
                your_deck=your_deck_sampled,
                your_prize=your_prize_sampled,
                opponent_deck=opp_deck_sampled,
                opponent_prize=opp_prize_sampled,
                opponent_hand=opp_hand_sampled,
                opponent_active=opp_active_sampled
            )
        except Exception:
            search_state = obs_dataclass

    if search_state is None:
        search_state = obs_dataclass

    root, sample = create_node(None, search_state, your_idx, your_deck, client, epoch=epoch)
    if not root.children:
        return [0], sample

    # AlphaZero Root Dirichlet Noise for self-play exploration (temperature > 0.01)
    if temperature > 0.01 and len(root.children) > 1:
        epsilon = 0.25
        alpha = 0.30
        try:
            dirichlet = torch.distributions.Dirichlet(torch.full((len(root.children),), alpha)).sample().tolist()
            for idx, child in enumerate(root.children):
                child.prob = (1.0 - epsilon) * child.prob + epsilon * dirichlet[idx]
        except Exception:
            pass


    # Pure AlphaZero MCTS Search Loop
    for _ in range(search_count):
        curr = root
        
        # 1. Selection Phase: Traverse tree using AlphaZero PUCT formula
        while curr.children and curr.children[0].node is not None:
            best_puct = -1e9
            best_child = curr.children[0]
            total_visits = sum(c.node.visit if c.node else 0 for c in curr.children) + 1
            
            for child in curr.children:
                if child.node is None:
                    continue
                # AlphaZero PUCT Equation
                q_val = child.node.total / max(1, child.node.visit)
                u_val = C_PUCT * child.prob * (math.sqrt(total_visits) / (1 + child.node.visit))
                puct = q_val + u_val
                if puct > best_puct:
                    best_puct = puct
                    best_child = child
            curr = best_child.node

        # 2. Expansion & Evaluation Phase
        if curr.children:
            unvisited_children = [c for c in curr.children if c.node is None]
            if unvisited_children:
                target_child = unvisited_children[0]
                next_state = curr.state
                try:
                    search_id = getattr(curr.state, "searchId", None)
                    if search_id is not None:
                        next_state = search_step(search_id, target_child.select)
                except Exception:
                    pass

                next_node, _ = create_node(curr, next_state, your_idx, your_deck, client, epoch=epoch)
                target_child.node = next_node
                
                # 3. Backpropagation Phase
                next_node.backprop(next_node.value)


    # 4. Action Selection based on Visit Count Distribution N(s, a)^(1/tau)
    visits = [c.node.visit if c.node else 0 for c in root.children]
    sum_visits = sum(visits)

    # Update training sample policy target to true MCTS visit count distribution pi_mcts(a) = N(a) / sum(N)
    if sample is not None and sum_visits > 0:
        sample.policy = [v / sum_visits for v in visits]

    if sum_visits == 0:
        return root.children[0].select, sample

    if temperature <= 0.01:
        # Deterministic max-visit action selection (Inference / Tournament Mode)
        best_idx = visits.index(max(visits))
        return root.children[best_idx].select, sample
    else:
        # Stochastic temperature-scaled action sampling (Exploration / Self-Play Mode)
        scaled_visits = [v ** (1.0 / temperature) for v in visits]
        total_sv = sum(scaled_visits)
        if total_sv <= 0:
            return root.children[0].select, sample
        probs = [v / total_sv for v in scaled_visits]
        selected_idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
        return root.children[selected_idx].select, sample



my_deck_global = None

def _load_my_deck():
    global my_deck_global
    if my_deck_global is None:
        candidate_paths = [
            "deck.csv",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deck.csv"),
            "/kaggle_simulations/agent/deck.csv",
            "deck_mewtwo_battle_cage.csv"
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8-sig") as f:
                        ids = [int(l.strip()) for l in f if l.strip()]
                        if len(ids) == 60:
                            my_deck_global = ids
                            break
                except Exception:
                    pass
        if my_deck_global is None:
            my_deck_global = OPPONENT_DECKS.get("Rulebasedmodel_Mewtwo", [])
    return my_deck_global


model_global = None

def agent(obs, configuration=None) -> list[int]:
    """Kaggle Competition Agent Entry Point."""
    global model_global

    your_deck = _load_my_deck()

    # Step 0: Initial deck registration step in Kaggle cabt
    # When obs["select"] is None, Kaggle expects the 60 card IDs list
    if isinstance(obs, dict):
        if obs.get("select") is None:
            return your_deck
    else:
        if getattr(obs, "select", None) is None:
            return your_deck

    if model_global is None:
        try:
            model_global = MyModel(
                MODEL_D_MODEL,
                MODEL_NUM_HEADS,
                MODEL_D_FEEDFORWARD,
                MODEL_NUM_LAYERS_ENCODER,
                MODEL_NUM_LAYERS_DECODER
            )
            model_path = "best_model.pth"
            if not os.path.exists(model_path):
                model_path = "model.pth"
            if os.path.exists(model_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model_global.load_state_dict(checkpoint["state_dict"], strict=False)
                else:
                    model_global.load_state_dict(checkpoint, strict=False)
                model_global.to(device)
                model_global.eval()
        except Exception:
            model_global = None

    remaining_time = 600.0
    if isinstance(obs, dict):
        remaining_time = float(obs.get('remainingOverageTime', 600.0))
    elif hasattr(obs, 'remainingOverageTime'):
        remaining_time = float(getattr(obs, 'remainingOverageTime', 600.0))

    IS_KAGGLE = os.path.exists('/kaggle_simulations/agent') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    if remaining_time < 60.0:
        search_count = 5
    elif remaining_time < 150.0:
        search_count = 10
    elif remaining_time < 300.0:
        search_count = 20 if IS_KAGGLE else 50
    else:
        search_count = 50 if IS_KAGGLE else 200

    try:
        selected, _ = mcts_agent(obs, your_deck, client=model_global, search_count=search_count, temperature=0.0)
        return selected
    except Exception:
        return [0]

