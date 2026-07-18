import glob
import math
import sys
import os
import torch
import torch.nn
import torch.nn.functional

# Resolve cg-lib path dynamically for Kaggle vs Local environments
try:
    cg_lib_path = glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0]
    sys.path.append(cg_lib_path)
except IndexError:
    pass

from cg.api import (
    AreaType,
    Card,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SelectContext,
    all_attack,
    all_card_data,
)

# Load card and attack metadata
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1

all_attack_list = all_attack()
attack_table = {a.attackId: a for a in all_attack_list}
attack_count = max(all_attack_list, key=lambda a: a.attackId).attackId + 1

# Model Architecture Hyperparameters (Single Source of Truth)
MODEL_D_MODEL = 256
MODEL_NUM_HEADS = 4
MODEL_D_FEEDFORWARD = 512
MODEL_NUM_LAYERS_ENCODER = 3
MODEL_NUM_LAYERS_DECODER = 1

num_words_encoder = 24
encoder_size = 22000

decoder_main_feature = 8
decoder_attack_offset = 14
decoder_card_offset = decoder_attack_offset + attack_count
decoder_size = decoder_card_offset + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count


def extract_card_features(card: Card | None) -> list[float]:
    features = [0.0] * 36  # 32 original + 4 energy-cost features
    if card is None:
        return features
    
    # 1. cardType (one-hot, 7 dims)
    if hasattr(card, "cardType") and 0 <= card.cardType <= 6:
        features[card.cardType] = 1.0
        
    # 2. energyType (one-hot, 11 dims)
    if hasattr(card, "energyType") and 0 <= card.energyType <= 10:
        features[7 + card.energyType] = 1.0
        
    # 3. HP (normalized, 1 dim)
    hp = getattr(card, "hp", 0)
    features[18] = min(1.0, hp / 400.0)
    
    # 4. Stage (one-hot, 3 dims)
    if getattr(card, "basic", False):
        features[19] = 1.0
    if getattr(card, "stage1", False):
        features[20] = 1.0
    if getattr(card, "stage2", False):
        features[21] = 1.0
        
    # 5. Subtype markers (4 dims)
    if getattr(card, "ex", False):
        features[22] = 1.0
    if getattr(card, "megaEx", False):
        features[23] = 1.0
    if getattr(card, "tera", False):
        features[24] = 1.0
    if getattr(card, "aceSpec", False):
        features[25] = 1.0
        
    # 6. Retreat Cost (normalized, 1 dim)
    retreat = getattr(card, "retreatCost", 0)
    features[26] = min(1.0, retreat / 4.0)
    
    # 7. Attacks & Abilities (5 dims)
    attacks = getattr(card, "attacks", None)
    if attacks:
        features[27] = 1.0
        features[28] = min(1.0, len(attacks) / 2.0)
        max_dmg = 0
        for aid in attacks:
            att = attack_table.get(aid)
            if att is not None:
                max_dmg = max(max_dmg, getattr(att, "damage", 0))
        features[29] = min(1.0, max_dmg / 300.0)
        
    skills = getattr(card, "skills", None)
    if skills:
        features[30] = 1.0
        features[31] = min(1.0, len(skills) / 2.0)

    # 8. Attack energy cost features (4 dims) — critical for energy management decisions
    # Encodes the cheapest attack's requirements by energy type, normalized by 4.
    # This allows the network to natively understand which energy is needed for each card.
    attacks = getattr(card, "attacks", None)
    if attacks:
        min_cost = 999
        cheapest_att = None
        for aid in attacks:
            att = attack_table.get(aid)
            if att is not None:
                cost = len(getattr(att, "energies", []))
                if cost < min_cost:
                    min_cost = cost
                    cheapest_att = att
        if cheapest_att is not None:
            energies = getattr(cheapest_att, "energies", [])
            features[32] = min(1.0, min_cost / 4.0)               # Total cost (normalized)
            features[33] = min(1.0, energies.count(1) / 4.0)      # Grass energy requirement
            features[34] = min(1.0, energies.count(5) / 4.0)      # Psychic energy requirement
            # Colorless/Special = any energy type not Grass or Psychic
            colorless = sum(1 for e in energies if e not in (1, 5))
            features[35] = min(1.0, colorless / 4.0)               # Colorless requirement
        
    return features


def get_encoder_card_map() -> torch.Tensor:
    card_map = [-1] * encoder_size
    pos = 0

    def safe_assign(idx, val):
        if 0 <= idx < encoder_size:
            card_map[idx] = val

    def add_pokemon_map(pos):
        for cid in range(card_count):
            safe_assign(pos + 2 + cid, cid)
            safe_assign(pos + 2 + card_count + cid, cid)
            safe_assign(pos + 2 + 2 * card_count + cid, cid)
        return pos + 2 + 3 * card_count

    for i in range(2):
        pos = add_pokemon_map(pos)

    for i in range(2):
        pos = add_pokemon_map(pos)

    for i in range(2):
        for cid in range(card_count):
            safe_assign(pos + 16 + cid, cid)
        pos += 16 + card_count

    for cid in range(card_count):
        safe_assign(pos + cid, cid)
    pos += card_count

    for cid in range(card_count):
        safe_assign(pos + cid, cid)
    pos += card_count

    for cid in range(card_count):
        safe_assign(pos + cid, cid)
    pos += card_count

    pos += 3
    return torch.tensor(card_map, dtype=torch.long)


def get_decoder_card_map() -> torch.Tensor:
    card_map = [-1] * decoder_size
    num_contexts = 1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION
    for offset_factor in range(num_contexts):
        base = decoder_card_offset + offset_factor * card_count
        for cid in range(card_count):
            if base + cid < decoder_size:
                card_map[base + cid] = cid
    return torch.tensor(card_map, dtype=torch.long)


class DecoderLayer(torch.nn.Module):
    """Decoder Layer of the Transformer Model."""
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super(DecoderLayer, self).__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.gelu(y)  # GELU is smoother than ReLU, better for transformers
        y = self.dropout(y)
        y = self.fc2(y)
        return self.norm2(res + y)


class MyModel(torch.nn.Module):
    """Transformer-like Model for Pokemon TCG Strategy and Action selection."""
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_feedforward: int,
                 num_layers_encoder: int,
                 num_layers_decoder: int
    ):
        super(MyModel, self).__init__()
        self.d_model = d_model

        self.encoder_bag = torch.nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout=0.1)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward))
        self.decoder_fc = torch.nn.Linear(d_model, 1)

        # Precompute static card features buffer
        features_list = []
        for cid in range(card_count):
            card = card_table.get(cid)
            features_list.append(extract_card_features(card))
        self.register_buffer("card_features", torch.tensor(features_list, dtype=torch.float32))

        # Register card maps as buffers
        self.register_buffer("encoder_card_map", get_encoder_card_map())
        self.register_buffer("decoder_card_map", get_decoder_card_map())

        # Feature projection layer: 36 dims (32 base + 4 energy-cost features)
        self.feature_projection = torch.nn.Linear(36, d_model)

    def forward(self,
                index_encoder: torch.Tensor,
                value_encoder: torch.Tensor,
                offset_encoder: torch.Tensor,
                index_decoder: torch.Tensor,
                value_decoder: torch.Tensor,
                offset_decoder: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = index_encoder.device

        # Project card features
        proj_features = self.feature_projection(self.card_features.to(device))  # (card_count, d_model)
        # Pad with zero features at index card_count for -1 mapping
        proj_features_padded = torch.cat([
            proj_features,
            torch.zeros(1, self.d_model, device=device)
        ], dim=0)

        # Map to vocabulary spaces
        enc_feat = proj_features_padded[self.encoder_card_map.to(device)]  # (encoder_size, d_model)
        dec_feat = proj_features_padded[self.decoder_card_map.to(device)]  # (decoder_size, d_model)

        # Combine learnable embeddings and card features
        enc_weight = self.encoder_bag.weight + enc_feat
        dec_weight = self.decoder_bag.weight + dec_feat

        # Use functional embedding_bag to pass the combined weights
        v = torch.nn.functional.embedding_bag(
            index_encoder, enc_weight, offset_encoder,
            per_sample_weights=value_encoder, mode="sum"
        )
        v = v.reshape(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.max(0).values)  # Max-pooling: better than mean at preserving high-signal tokens

        p = torch.nn.functional.embedding_bag(
            index_decoder, dec_weight, offset_decoder,
            per_sample_weights=value_decoder, mode="sum"
        )
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        # Policy outputs raw logits — softmax applied in cross-entropy loss or at inference
        return (v, p)


class SparseVector:
    """Helper class to construct sparse representations for EmbeddingBag inputs."""
    index: list[int]
    value: list[float]
    offset: list[int]
    pos: int

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []
        self.pos = 0

    def add(self, index: int, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))


class LearnInput:
    """Helper class to construct batch inputs for the neural network."""
    index: list[int]
    value: list[float]
    offset: list[int]

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []

    def add(self, sv: SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)


def add_card(sv: SparseVector, card: Card | Pokemon | None):
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(card_count)


def add_cards(sv: SparseVector, cards: list[Card] | None, value: float):
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)


def add_pokemon(sv: SparseVector, poke: Pokemon | None):
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)


def add_player(sv: SparseVector, ps: PlayerState):
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25)


def get_encoder_input(obs: Observation, your_deck: list[int]) -> SparseVector:
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):  # For bench
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:  # Not last
                sv.pos = pos  # Return to the previous position
    
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if len(ps.active) > 0:
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)
        
    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)
        
    sv.word_start()
    for id in your_deck:
        sv.add(id, 0.25)
    sv.add_pos(card_count)
        
    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    # Explicit prize differential: positive = we are ahead, negative = opponent is ahead
    my_prizes = len(state.players[your_index].prize)
    opp_prizes = len(state.players[1 - your_index].prize)
    sv.add_single((opp_prizes - my_prizes) / 6.0)
    return sv


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def decoder_main(sv: SparseVector, feature_index: int, card: Card | Pokemon | None):
    if card is not None:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)


def decoder_card_id(sv: SparseVector, context: SelectContext, card_id: int):
    sv.add(decoder_card_offset + (decoder_main_feature + context) * card_count + card_id, 1)


def decoder_card(sv: SparseVector, context: SelectContext, card: Card | Pokemon | None):
    if card is not None:
        decoder_card_id(sv, context, card.id)


def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    for action in actions:
        sv.word_start()
        
        if len(action) == 0:
            sv.add(0, 1)
            continue
        
        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END:
                    sv.add(1, 1)
                case OptionType.YES:
                    sv.add(2, 1)
                case OptionType.NO:
                    sv.add(3, 1)
                case OptionType.SPECIAL_CONDITION:
                    sv.add(4 + o.specialConditionType, 1)
                case OptionType.NUMBER:
                    sv.add(9 + min(o.number, 4), 1)
                case OptionType.ATTACK:
                    sv.add(decoder_attack_offset + o.attackId, 1)
                    if len(ps.active) > 0 and ps.active[0] is not None:
                        decoder_main(sv, 7, ps.active[0])
                case OptionType.PLAY:
                    decoder_main(sv, 0, ps.hand[o.index])
                case OptionType.ATTACH:
                    decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.EVOLVE:
                    decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.ABILITY:
                    decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
                case OptionType.DISCARD:
                    decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
                case OptionType.RETREAT:
                    decoder_main(sv, 7, ps.active[0])
                case OptionType.CARD:
                    decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex))
                case OptionType.TOOL_CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    tools = card.tools if (card is not None and card.tools is not None) else []
                    tool_card = tools[o.toolIndex] if o.toolIndex < len(tools) else None
                    decoder_card(sv, context, tool_card)
                case OptionType.ENERGY_CARD | OptionType.ENERGY:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    energies = card.energyCards if (card is not None and card.energyCards is not None) else []
                    energy_card = energies[o.energyIndex] if o.energyIndex < len(energies) else None
                    decoder_card(sv, context, energy_card)
                case OptionType.SKILL:
                    decoder_card_id(sv, context, o.cardId)

    return sv


def eval_nn(sv_enc: SparseVector, sv_dec: SparseVector, model) -> tuple[float, list[float]]:
    if hasattr(model, "eval_nn"):
        return model.eval_nn(sv_enc, sv_dec)
        
    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device))

    return (value.tolist()[0][0], policy.tolist()[0])
