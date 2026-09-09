import glob
import math
import os
import random
import sys
import torch

try:
    from model import (
        MyModel,
        SparseVector,
        get_encoder_input,
        get_decoder_input,
        eval_nn,
        card_table,
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
        card_table,
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
    CardType,
    to_observation_class,
    search_begin,
    search_step,
    search_end,
)

SEARCH_COUNT = 200  # Default MCTS Search count (≥200 for deep MCTS exploration)

try:
    from expert_knowledge import USE_EXPERT_GUIDANCE, EXPERT_WEIGHT, get_expert_bonus
except ImportError:
    try:
        from src.expert_knowledge import USE_EXPERT_GUIDANCE, EXPERT_WEIGHT, get_expert_bonus
    except ImportError:
        USE_EXPERT_GUIDANCE = False
        EXPERT_WEIGHT = 0.0
        def get_expert_bonus(*args, **kwargs):
            return 0.0, ""


class LearnSample:
    """Single Training Sample collected during gameplay/self-play."""
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec: SparseVector):
        self.value = value        # Encoder output (expected outcome)
        self.policy = policy      # Decoder output (action probability distribution)
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec
        self.nn_top1_idx: int = 0   # NN top-1 action BEFORE expert/MCTS (for disagreement logging)


class GPUInferenceClient:
    """Client wrapper for worker threads/processes to send inference queries to a GPU batch server."""
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
        self.value = 0.0
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
                model: MyModel | GPUInferenceClient,
                temperature: float = 1.0,
                opponent_name: str = ""
) -> tuple[Node, LearnSample | None]:
    """Create a new node and evaluate its state using the neural network and expert guidance."""
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
        # Enumerate potential action combinations
        actions = []
        indices = list(range(obs.select.maxCount))
        for _ in range(64):
            actions.append(indices.copy())
            for i in range(len(indices)):
                index = len(indices) - i - 1
                if indices[index] < len(obs.select.option) - i - 1:
                    indices[index] += 1
                    for j in range(index + 1, len(indices)):
                        indices[j] = indices[j - 1] + 1
                    break
            else:
                break
            
        sv_enc = get_encoder_input(obs, your_deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn(sv_enc, sv_dec, model)
        v = value
        if state.yourIndex != your_index:
            v = -v
        node.value = v
        node.backprop(v)

        # Compute NN Policy Entropy for Dynamic Expert Scaling
        nn_entropy = 0.5
        if len(policy) > 1:
            try:
                max_p = max(policy)
                exp_probs = [math.exp(p - max_p) for p in policy]
                sum_p = sum(exp_probs)
                probs = [p / sum_p for p in exp_probs] if sum_p > 0 else [1.0 / len(policy)] * len(policy)
                h_raw = -sum(p * math.log(max(1e-9, p)) for p in probs)
                nn_entropy = h_raw / math.log(len(policy))  # Normalized entropy [0, 1]
            except Exception:
                nn_entropy = 0.5

        # Capture NN top-1 action BEFORE expert modification (for disagreement logging)
        nn_top1_idx = policy.index(max(policy)) if policy else 0

        # Apply expert guidance prior bias to policy logits before softmax.
        # Scale = 8.0: expert bonus range [-0.90, +0.50] × EXPERT_WEIGHT(0.25) × confidence
        # maps to logit adjustments of [-1.8, +1.0], which is meaningful but cannot
        # fully override a strong NN logit difference (NN range typically [-3, +3]).
        if USE_EXPERT_GUIDANCE:
            try:
                options = getattr(obs.select, "option", [])
                if isinstance(options, (list, tuple)) and len(options) > 0:
                    for i in range(len(policy)):
                        if actions[i] and len(actions[i]) > 0:
                            opt_idx = actions[i][0]
                            if 0 <= opt_idx < len(options):
                                opt = options[opt_idx]
                                exp_bonus, _ = get_expert_bonus(obs, opt, opponent_name=opponent_name, nn_entropy=nn_entropy)
                                # Scale 8.0: calibrated so expert guides but cannot dominate a confident NN
                                policy[i] += exp_bonus * 8.0
            except Exception:
                pass

        # Numerically stable softmax with temperature scaling
        prob_sum = 0.0
        max_p = max(policy) if len(policy) > 0 else 0.0
        scale = (1.0 / max(0.05, temperature)) if temperature > 0.0 else 100.0

        for i in range(len(policy)):
            p = math.exp((policy[i] - max_p) * scale)
            node.children.append(Child(actions[i], p))
            prob_sum += p

        if prob_sum > 0.0:
            for c in node.children:
                c.prob /= prob_sum
        else:
            uniform = 1.0 / max(1, len(node.children))
            for c in node.children:
                c.prob = uniform

        # AlphaZero Standard: Add Dirichlet Exploration Noise at Root Node during self-play training
        if parent is None and temperature > 0.0 and len(node.children) > 1:
            try:
                import numpy as np
                alpha = 0.30
                epsilon = 0.25
                dirichlet_noise = np.random.dirichlet([alpha] * len(node.children))
                for idx_c, child_node in enumerate(node.children):
                    child_node.prob = (1.0 - epsilon) * child_node.prob + epsilon * float(dirichlet_noise[idx_c])
            except Exception:
                pass

        sample = LearnSample(value, policy, sv_enc, sv_dec)
        # Store pre-MCTS NN top-1 for disagreement logging in worker.py
        sample.nn_top1_idx = nn_top1_idx

    return (node, sample)


def mcts_agent(obs_dict: dict,
               your_deck: list[int],
               model: MyModel | GPUInferenceClient,
               search_count: int = SEARCH_COUNT,
               temperature: float = 1.0,
               **kwargs
) -> tuple[list[int], LearnSample]:
    """Perform MCTS exploration and select the best action list, returning it and a training sample."""
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    opp_name = kwargs.get("opponent_name", "")
    active = state.players[1 - your_index].active

    opp_idx = 1 - your_index
    opp_deck_cnt = state.players[opp_idx].deckCount
    opp_prize_cnt = len(state.players[opp_idx].prize)
    opp_hand_cnt = state.players[opp_idx].handCount

    # Find valid Pokémon cards from your_deck pool for facedown active prediction
    poke_in_deck = [cid for cid in your_deck if cid in card_table and card_table[cid].cardType == CardType.POKEMON]
    basic_poke_in_deck = [cid for cid in poke_in_deck if getattr(card_table[cid], "basic", False)]
    default_active_id = (basic_poke_in_deck[0] if basic_poke_in_deck 
                         else (poke_in_deck[0] if poke_in_deck else 400))

    # Dynamically sample unknown opponent cards from your_deck pool
    opp_deck = random.choices(your_deck, k=opp_deck_cnt) if opp_deck_cnt > 0 else []
    opp_prize = random.choices(your_deck, k=opp_prize_cnt) if opp_prize_cnt > 0 else []
    opp_hand = random.choices(your_deck, k=opp_hand_cnt) if opp_hand_cnt > 0 else []
    opp_active = [default_active_id] if (len(active) > 0 and active[0] is None) else []
    
    # Construct belief states of hidden information for planning
    search_state = search_begin(
        obs,
        your_deck=random.sample(your_deck, state.players[your_index].deckCount),
        your_prize=random.sample(your_deck, len(state.players[your_index].prize)),
        opponent_deck=opp_deck,
        opponent_prize=opp_prize,
        opponent_hand=opp_hand,
        opponent_active=opp_active
    )
    
    try:
        root, sample = create_node(None, search_state, your_index, your_deck, model, temperature=temperature, opponent_name=opp_name)

        # Search loop
        for _ in range(search_count):
            current = root
            while True:
                value = -1e9
                c = 0.4 * math.sqrt(current.visit)
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
                
                if next_child is None:
                    break

                if next_child.node is None:
                    try:
                        search_state_step = search_step(current.state.searchId, next_child.select)
                        next_child.node, _ = create_node(current, search_state_step, your_index, your_deck, model, temperature=temperature, opponent_name=opp_name)
                    except Exception:
                        # Safely terminate tree expansion if search_step fails on a branch (e.g. mulligan/setup edge case)
                        break
                    break
                else:
                    current = next_child.node
                    if current.state.observation.current.result >= 0:
                        current.backprop(current.value)
                        break

        # Select the child with the highest visit count
        max_child = None
        max_visit = -1
        min_value = 1.0
        for child in root.children:
            if child.node is not None:
                if max_visit < child.node.visit:
                    max_child = child
                    max_visit = child.node.visit
                v = child.node.total / child.node.visit
                if min_value > v:
                    min_value = v

        if max_child is None:
            if root.children:
                best_c = max(root.children, key=lambda c: c.prob)
                return (best_c.select, sample)
            return ([0], sample)

        # Generate targets/labels for training policy (AlphaZero MCTS visit count distribution)
        if sample is not None:
            sample.value = root.total / max(1, root.visit)
            total_child_visits = sum(c.node.visit if c.node is not None else 0 for c in root.children)
            if total_child_visits > 0:
                for i, child in enumerate(root.children):
                    sample.policy[i] = (child.node.visit / total_child_visits) if child.node is not None else 0.0
            else:
                uniform_p = 1.0 / max(1, len(root.children))
                for i in range(len(root.children)):
                    sample.policy[i] = uniform_p

        return (max_child.select, sample)
    finally:
        try:
            search_end()
        except Exception:
            pass


def random_agent(obs_dict: dict) -> list[int]:
    """Random baseline agent."""
    obs = to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)


# Cached global model and deck to avoid repeated loading
_model = None
_deck = None

# Default TR Mewtwo ex deck (60 cards)
MEWTWO_DECK = [
    1, 1, 1, 1, 1, 1, 1, 1, 15, 15, 15, 15,
    400, 400, 400, 400, 401, 401, 401, 401, 414, 414, 431, 431, 434, 434,
    1086, 1094, 1094, 1094, 1097, 1097, 1121, 1121, 1134, 1134, 1134, 1134,
    1152, 1152, 1152, 1152, 1159, 1175, 1175, 1216, 1216, 1216, 1216, 1218, 1218,
    1219, 1220, 1220, 1227, 1227, 1227, 1264, 1264, 1264
]


def agent(obs_dict: dict) -> list[int]:
    """Kaggle Competition Entry point function."""
    global _model, _deck
    
    # Load deck list
    if _deck is None:
        possible_paths = []
        if "__file__" in globals():
            base_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths.append(os.path.join(base_dir, "deck.csv"))
            possible_paths.append(os.path.join(os.path.dirname(base_dir), "deck.csv"))
        possible_paths.append(os.path.join(os.getcwd(), "deck.csv"))
        possible_paths.append("/kaggle_simulations/agent/deck.csv")

        for d_path in possible_paths:
            if os.path.exists(d_path):
                try:
                    with open(d_path, "r") as f:
                        loaded = [int(line.strip()) for line in f if line.strip()]
                    if len(loaded) == 60:
                        _deck = loaded
                        break
                except Exception:
                    pass
        
        if _deck is None:
            _deck = MEWTWO_DECK.copy()

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
            MODEL_NUM_LAYERS_DECODER,
        )

        possible_model_paths = []
        if "__file__" in globals():
            base_dir = os.path.dirname(os.path.abspath(__file__))
            possible_model_paths.append(os.path.join(base_dir, "best_model.pth"))
            possible_model_paths.append(os.path.join(base_dir, "model.pth"))
            possible_model_paths.append(os.path.join(os.path.dirname(base_dir), "best_model.pth"))
            possible_model_paths.append(os.path.join(os.path.dirname(base_dir), "model.pth"))
        possible_model_paths.append(os.path.join(os.getcwd(), "best_model.pth"))
        possible_model_paths.append(os.path.join(os.getcwd(), "model.pth"))
        possible_model_paths.append("/kaggle_simulations/agent/best_model.pth")
        possible_model_paths.append("/kaggle_simulations/agent/model.pth")

        model_path = None
        for m_path in possible_model_paths:
            if os.path.exists(m_path):
                model_path = m_path
                break

        if model_path and os.path.exists(model_path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                _model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                _model.load_state_dict(checkpoint, strict=False)

        _model = _model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        _model.eval()

    # Kaggle Submission Entrypoint: search_count = 0 for ultra-fast instant NN + Expert Prior inference (<0.005s per decision, zero timeout risk)
    search_count = 0

    with torch.inference_mode():
        action, _ = mcts_agent(obs_dict, _deck, _model, search_count=search_count, temperature=0.0)
        
    return action