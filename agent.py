import math
import random
import torch
from cg.api import (
    Observation,
    SearchState,
    to_observation_class,
    search_begin,
    search_step,
    search_end,
)
from encoders import SparseVector, get_encoder_input, get_decoder_input
from models import MyModel

class LearnSample:
    """
    A single training sample containing state representation, action policy targets,
    and board value.
    """
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec: SparseVector):
        self.value = value
        self.policy = policy
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec

class Child:
    """
    Child node representation in MCTS tree.
    """
    node: 'Node | None'
    select: list[int]
    prob: float

    def __init__(self, select: list[int], prob: float):
        self.node = None
        self.select = select
        self.prob = prob

class Node:
    """
    Node in the MCTS tree.
    """
    value: float
    total: float
    visit: int
    parent: 'Node | None'
    children: list[Child]
    state: SearchState

    def __init__(self, parent: 'Node | None', state: SearchState):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    def backprop(self, value: float):
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)

def eval_nn(sv_enc: SparseVector, sv_dec: SparseVector, model: MyModel) -> tuple[float, list[float]]:
    """
    Evaluates the model on a state representation and a list of actions.
    """
    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device)
    )
    return (value.tolist()[0][0], policy.tolist()[0])

def create_node(parent: Node | None,
                search_state: SearchState,
                your_index: int,
                your_deck: list[int],
                model: MyModel
) -> tuple[Node, LearnSample | None]:
    """
    Creates a new Node and initializes its children using policy predictions from the network.
    """
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
        # Generate legal actions combinations (up to 64)
        actions = []
        indices = list(range(obs.select.maxCount))
        for _ in range(64):
            actions.append(indices.copy())
            for i in range(len(indices)):
                idx = len(indices) - i - 1
                if indices[idx] < len(obs.select.option) - i - 1:
                    indices[idx] += 1
                    for j in range(idx + 1, len(indices)):
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

        total_p = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * 10.0)
            node.children.append(Child(actions[i], p))
            total_p += p
        for c in node.children:
            c.prob /= total_p
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)

def mcts_agent(obs_dict: dict, your_deck: list[int], model: MyModel, search_count: int = 10) -> tuple[list[int], LearnSample]:
    """
    Monte Carlo Tree Search agent.
    """
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active

    # Randomly sample cards to fill hidden info in search environment
    search_state = search_begin(
        obs,
        your_deck=random.sample(your_deck, state.players[your_index].deckCount),
        your_prize=random.sample(your_deck, len(state.players[your_index].prize)),
        opponent_deck=[1072] * state.players[1 - your_index].deckCount,  # Snorlax ID
        opponent_prize=[1] * len(state.players[1 - your_index].prize),   # Basic Energy ID
        opponent_hand=[1] * state.players[1 - your_index].handCount,    # Basic Energy ID
        opponent_active=[1072] if len(active) > 0 and active[0] is None else []
    )
    root, sample = create_node(None, search_state, your_index, your_deck, model)

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
                step_state = search_step(current.state.searchId, next_child.select)
                next_child.node, _ = create_node(current, step_state, your_index, your_deck, model)
                break
            else:
                current = next_child.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

    # Select child with most visits
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

    # Generate training target values
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
    """
    Agent that selects moves randomly. Used as an evaluation baseline.
    """
    obs = to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)
