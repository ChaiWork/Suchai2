import glob
import math
import os
import random
import sys
import torch

from model import (
    MyModel,
    SparseVector,
    get_encoder_input,
    get_decoder_input,
    eval_nn,
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
)

SEARCH_COUNT = 10  # MCTS Search count


class LearnSample:
    """Single Training Sample collected during gameplay/self-play."""
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec: SparseVector):
        self.value = value        # Encoder output (expected outcome)
        self.policy = policy      # Decoder output (action probability distribution)
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec


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
        # Enumerate up to 64 potential action combinations
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

        # Compute probabilities using softmax-like temperature scaling
        prob_sum = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * 10.0)
            node.children.append(Child(actions[i], p))
            prob_sum += p
        for c in node.children:
            c.prob /= prob_sum
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)


def mcts_agent(obs_dict: dict, your_deck: list[int], model: MyModel, search_count: int = SEARCH_COUNT) -> tuple[list[int], LearnSample]:
    """Perform MCTS exploration and select the best action list, returning it and a training sample."""
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    
    # Construct belief states of hidden information for planning
    search_state = search_begin(
        obs,
        your_deck=random.sample(your_deck, state.players[your_index].deckCount),
        your_prize=random.sample(your_deck, len(state.players[your_index].prize)),
        opponent_deck=[1072] * state.players[1 - your_index].deckCount,          # Snorlax card ID
        opponent_prize=[1] * len(state.players[1 - your_index].prize),           # Basic Energy
        opponent_hand=[1] * state.players[1 - your_index].handCount,             # Basic Energy
        opponent_active=[1072] if len(active) > 0 and active[0] is None else []   # Snorlax if facedown
    )
    
    root, sample = create_node(None, search_state, your_index, your_deck, model)

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

    # Generate targets/labels for training policy
    sample.value = root.total / root.visit
    for i in range(len(root.children)):
        child = root.children[i]
        v = sample.value
        if child.node is None:
            v = min_value - v - 0.03
        else:
            v = child.node.total / child.node.visit - v
        sample.policy[i] = max(-1.0, min(1.0, v))

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
        base_path = os.path.dirname(os.path.abspath(__file__))
        deck_path = os.path.join(base_path, "deck.csv")
        try:
            with open(deck_path, "r") as f:
                _deck = [int(line.strip()) for line in f if line.strip()]
        except Exception:
            # Default sample deck
            _deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

    # Load model weights
    if _model is None:
        _model = MyModel(128, 2, 256, 1, 1)
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "model.pth")
        
        # Load weights if available
        if os.path.exists(model_path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model.load_state_dict(torch.load(model_path, map_location=device))
        
        _model = _model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        _model.eval()

    with torch.inference_mode():
        action, _ = mcts_agent(obs_dict, _deck, _model)
        
    return action
