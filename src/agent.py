import glob
import math
import os
import random
import sys
import torch

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
)

SEARCH_COUNT = 50  # MCTS Search count — need ≥50 for meaningful visit differentiation


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
        self.conn.send((sv_enc, sv_dec))
        return self.conn.recv()


class Child:
    """MCTS Node Child representing a possible selected action combination."""
    node: 'Node | None'
    select: list[int]  # Selected option indices
    prob: float        # Probability of selection

    def __init__(self, select: list[int], prob: float):
        self.node = None
        self.select = select
        self.prob = prob


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


def create_node(parent: Node | None,
                search_state: SearchState,
                your_index: int,
                your_deck: list[int],
                model: MyModel
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
        node.backprop(node.value)
        sample = None
    else:
        # Enumerate up to 128 potential action combinations prioritizing high-value options
        actions = []
        options = obs.select.option
        
        high_priority = []
        low_priority = []
        
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
        
        for idx, opt in enumerate(options):
            if opt.type in HIGH_PRIORITY_TYPES:
                high_priority.append(idx)
            else:
                low_priority.append(idx)
                
        sorted_indices = high_priority + low_priority
        n = len(sorted_indices)
        k = obs.select.maxCount
        
        # Generate combinations using sorted_indices
        if k <= n:
            comb_positions = list(range(k))
            for _ in range(128):
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
        if len(options) > 128 and len(actions) == 128:
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
        node.backprop(v)

        # Convert raw logits to probabilities via numerically stable softmax
        max_logit = max(policy)
        prob_sum = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] - max_logit)  # Numerically stable softmax
            node.children.append(Child(actions[i], p))
            prob_sum += p
        for c in node.children:
            c.prob /= prob_sum
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)

# --- Opponent Deck Database & Belief State Identification ---
OPPONENT_DECKS = {
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


def identify_opponent_deck(revealed_ids: list[int], opponent_decks: dict) -> list[int]:
    """Identifies the opponent's deck template using multiset intersection match."""
    if not revealed_ids:
        # Default to BasicallyBot if no cards have been revealed yet
        return opponent_decks.get("BasicallyBot_85134910")
        
    best_name = None
    best_matches = -1
    
    for name, deck in opponent_decks.items():
        # Count frequency of each card ID in the deck
        deck_counts = {}
        for cid in deck:
            deck_counts[cid] = deck_counts.get(cid, 0) + 1
            
        # Count matches based on multiset intersection
        matches = 0
        for cid in revealed_ids:
            if deck_counts.get(cid, 0) > 0:
                matches += 1
                deck_counts[cid] -= 1
                
        if matches > best_matches:
            best_matches = matches
            best_name = name
            
    return opponent_decks[best_name]


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


def mcts_agent(obs_dict: dict, your_deck: list[int], model: MyModel, search_count: int = None) -> tuple[list[int], LearnSample]:
    """Perform MCTS exploration and select the best action list, returning it and a training sample."""
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    
    # Dynamically match opponent deck based on revealed cards
    opp_index = 1 - your_index
    revealed_ids = get_opponent_revealed_card_ids(obs, opp_index)
    matched_deck = identify_opponent_deck(revealed_ids, OPPONENT_DECKS)
    
    remaining_cards = matched_deck.copy()
    for cid in revealed_ids:
        if cid in remaining_cards:
            remaining_cards.remove(cid)
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
    
    # Sample own hidden zones correctly
    own_remaining = your_deck.copy()
    own_visible = get_own_visible_card_ids(obs, your_index)
    for cid in own_visible:
        if cid in own_remaining:
            own_remaining.remove(cid)
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
    
    root, sample = create_node(None, search_state, your_index, your_deck, model)

    # Add Dirichlet noise to root prior for exploration (AlphaZero-style)
    # Without noise, MCTS always explores the same paths from the NN prior,
    # causing mode collapse (the agent repeatedly selects "end turn").
    if len(root.children) > 0:
        dir_alpha = 0.3  # TCG has moderate action space
        turn = state.turn if (state is not None) else 0
        noise_frac = 0.40
        noise = [random.gammavariate(dir_alpha, 1.0) for _ in root.children]
        noise_sum = sum(noise) + 1e-8
        noise = [n / noise_sum for n in noise]
        for i, child in enumerate(root.children):
            child.prob = (1.0 - noise_frac) * child.prob + noise_frac * noise[i]

    # Dynamic Simulation Count based on branch branching factor
    if search_count is None:
        dynamic_search_count = SEARCH_COUNT
    else:
        dynamic_search_count = search_count

    # Caps simulations in the Kaggle runtime environment to avoid TIMEOUT (no GPU)
    IS_KAGGLE = os.path.exists('/kaggle_simulations/agent') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    if IS_KAGGLE:
        dynamic_search_count = min(dynamic_search_count, 15)

    # Search loop
    for _ in range(dynamic_search_count):
        current = root
        while True:
            value = -1e9
            # Dynamic PUCT Exploration
            c = 1.25 * math.sqrt(current.visit)
            next_child = None
            for child in current.children:
                visit = 0
                if child.node is None:
                    v = current.total / current.visit
                else:
                    v = child.node.total / child.node.visit
                    visit = child.node.visit
                
                if current.state.observation.current.yourIndex != your_index:
                    v = -v
                v += c * child.prob / (1 + visit)
                if value < v:
                    value = v
                    next_child = child
            
            if next_child.node is None:
                search_state = search_step(current.state.searchId, next_child.select)
                next_child.node, _ = create_node(current, search_state, your_index, your_deck, model)
                break
            else:
                current = next_child.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

    # Select the child with the highest visit count
    max_child = None
    max_visit = -1
    min_value = 10.0
    for child in root.children:
        if child.node is not None:
            if max_visit < child.node.visit:
                max_child = child
                max_visit = child.node.visit
            v = child.node.total / child.node.visit
            if min_value > v:
                min_value = v

    # Generate targets/labels for training
    # Value target: root mean value
    sample.value = root.total / root.visit

    # Policy target: visit count distribution (AlphaZero-style)
    # Visit counts are more robust than Q-value differences with limited simulations.
    total_child_visits = sum(
        child.node.visit for child in root.children if child.node is not None
    )
    if total_child_visits > 0:
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
    return (max_child.select, sample)


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
            # Default sample deck
            _deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

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
        model_path = os.path.join(base_path, "model.pth")
        
        # Load weights if available
        if os.path.exists(model_path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                _model.load_state_dict(checkpoint["state_dict"])
            else:
                _model.load_state_dict(checkpoint)
        
        _model = _model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        _model.eval()

    # Determine adaptive search count budget based on current turn
    obs = to_observation_class(obs_dict)
    turn = obs.current.turn if (obs.current is not None) else 0
    
    IS_KAGGLE = os.path.exists('/kaggle_simulations/agent') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    if IS_KAGGLE:
        # Lower budget on Kaggle to prevent TIMEOUT on weak CPU
        if turn <= 3:
            search_count = 20
        elif turn <= 8:
            search_count = 15
        else:
            search_count = 10
    else:
        # Standard local/eval budget
        if turn <= 3:
            search_count = 150
        elif turn <= 8:
            search_count = 100
        else:
            search_count = 50

    with torch.inference_mode():
        action, _ = mcts_agent(obs_dict, _deck, _model, search_count=search_count)
        
    return action
