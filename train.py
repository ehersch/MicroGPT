import random
from re import L
from datasets import load_dataset
from gpt_model import Tokenizer
from dataclasses import dataclass
from autograd import Value
from utils import softmax
import math

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("zhyncs/sonnet")

tokenizer = Tokenizer(ds["train"])
"""
First define model architecture
"""


@dataclass
class Config:
    embed_dim: int
    attn_heads: int
    layers: int
    block_size: int
    vocab_size: int

    @property
    def hidden_dim(self):
        return self.embed_dim // self.attn_heads


config = Config(
    embed_dim=16,
    attn_heads=4,
    layers=1,
    block_size=16,
    vocab_size=len(tokenizer.vocab),
)


def create_matrix(in_dim, out_dim, std=0.08):
    # creates a matrix of size in_dim x out_dim
    return [
        [Value(random.gauss(0, std)) for _ in range(out_dim)] for _ in range(in_dim)
    ]


state_dict = {}
for i in range(config.layers):
    # Projects embedding -> Query vectors
    state_dict[f"layer_{i}_attn_wq"] = create_matrix(config.embed_dim, config.embed_dim)
    # Projects embedding -> Key vectors
    state_dict[f"layer_{i}_attn_wk"] = create_matrix(config.embed_dim, config.embed_dim)
    # Projects embedding -> Value vectors
    state_dict[f"layer_{i}_attn_wv"] = create_matrix(config.embed_dim, config.embed_dim)
    # Final projection after concatenating all heads.
    state_dict[f"layer_{i}_attn_wo"] = create_matrix(config.embed_dim, config.embed_dim)
    # Standard MLP: Linear(embed_dim -> 4*embed_dim); GELU; Linear(4*embed_dim -> embed_dim)
    state_dict[f"layer_{i}_fc1"] = create_matrix(config.embed_dim, 4 * config.embed_dim)
    state_dict[f"layer_{i}_fc2"] = create_matrix(4 * config.embed_dim, config.embed_dim)

# Used to map final hidden state -> logits over vocabulary:
state_dict[f"output_head"] = create_matrix(config.vocab_size, config.embed_dim)
# This is positional embedding
state_dict[f"pos_embedding"] = create_matrix(config.block_size, config.embed_dim)
# this is token embedding
state_dict[f"token_embedding"] = create_matrix(config.vocab_size, config.embed_dim)

params = [p for mat in state_dict.values() for row in mat for p in row]


def train(model, data, steps=1000):
    lr = 0.01
    beta_1 = 0.85
    beta_2 = 0.99
    eps_adam = 1e-8

    m = [0 for _ in range(len(params))]
    v = [0 for _ in range(len(params))]
    for step in steps:
        # Make a single AdamW step

        sonnet = data[step]["text"]
        tokens = [tokenizer[c] for c in list(sonnet)]
        n = min(config.block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(config.layers)], [
            [] for _ in range(config.layers)
        ]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = model.forward(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss = -probs[target_id].log()  # - log likelihood

            losses.append(loss)

        loss = sum(losses, Value(0.0)) / Value(float(n))
        loss.backward()

        for i, p in enumerate(params):
            g_t = p.grad
            m[i] = beta_1 * m[i - 1] * (1 - beta_1) * g_t
            v[i] = beta_2 * m[i - 1] * (1 - beta_2) * g_t**2
            lr_t = lr * math.sqrt(1 - beta_2**i) / (1 - beta_1**i)

            p.val -= lr_t * m[i] / (v[i] ** (0.5) + eps_adam)

            p.grad = 0

        print(f"step {step+1:4d} / {steps:4d} | loss {loss.val:.4f}")
