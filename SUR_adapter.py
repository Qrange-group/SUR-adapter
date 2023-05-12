import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size=768):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = value

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32)
        )
        attention_weights = nn.functional.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V) + value
        result = self.output_layer(weighted_values) + weighted_values

        return result


class Adapter(nn.Module):
    def __init__(self, depth=2, adapter_weight=0.01, sd_text_size=768):
        super(Adapter, self).__init__()

        self.adapter_weight = adapter_weight
        
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=sd_text_size, nhead=8, dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=depth
        )

        # Attension
        self.attention = Attention(hidden_size=sd_text_size)

        # LLM layer
        self.fc = nn.Linear(sd_text_size, sd_text_size)
        nn.init.zeros_(self.fc.weight)

    def forward(self, x):
        out_transformer_encoder = self.transformer_encoder(x)
        out_attention = self.attention(query=out_transformer_encoder, key=x, value=x)
        out_llm = self.fc(out_attention)
        out = self.adapter_weight * out_llm + (1 - self.adapter_weight) * x

        return out, out_transformer_encoder, out_llm
