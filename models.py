import torch
import torch.nn
import torch.nn.functional
from cg.api import all_card_data, all_attack, SelectContext

# Calculate dimensions based on card data
all_card = all_card_data()
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1
attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1

NUM_WORDS_ENCODER = 24
ENCODER_SIZE = 22000

DECODER_MAIN_FEATURE = 8
DECODER_ATTACK_OFFSET = 14
DECODER_CARD_OFFSET = DECODER_ATTACK_OFFSET + attack_count
DECODER_SIZE = DECODER_CARD_OFFSET + (1 + DECODER_MAIN_FEATURE + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count

class DecoderLayer(torch.nn.Module):
    """
    Decoder Layer of MyModel querying encoder outputs against action representations
    using MultiheadAttention.
    """
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super(DecoderLayer, self).__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)

class MyModel(torch.nn.Module):
    """
    Transformer-based Actor-Critic model for Pokémon TCG.
    Estimates the board state value (advantage/win rate) and scores the available legal actions.
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_feedforward: int,
                 num_layers_encoder: int,
                 num_layers_decoder: int
    ):
        super(MyModel, self).__init__()
        self.d_model = d_model

        self.encoder_bag = torch.nn.EmbeddingBag(ENCODER_SIZE, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, dropout=0)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)

        self.decoder_bag = torch.nn.EmbeddingBag(DECODER_SIZE, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward))
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    def forward(self,
                index_encoder: torch.Tensor,
                value_encoder: torch.Tensor,
                offset_encoder: torch.Tensor,
                index_decoder: torch.Tensor,
                value_decoder: torch.Tensor,
                offset_decoder: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Process encoder inputs (state description)
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, NUM_WORDS_ENCODER, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        # Process decoder inputs (actions)
        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        p = torch.tanh(p)
        return (v, p)
