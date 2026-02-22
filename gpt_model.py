import math
from autograd import Value
from utils import linear, matmul, transpose, softmax, rmsnorm
import random


class Tokenizer:
    def __init__(self, train_ds):
        """
        Defines a byte tokenizer for efficient tokenization
        """

        unique_chars = set()

        for row in train_ds:
            chars_list = list(row["text"])
            for c in chars_list:
                unique_chars.add(c)

        self.vocab = {index: char for index, char in enumerate(unique_chars)}

        self.tokenizer = {char: index for index, char in enumerate(unique_chars)}

    def tokenize(self, text):
        """
        Tokenize some text.
        """
        return [self.tokenizer[c] for c in list(text)]

    def get_text(self, token_ids):
        """
        Go from list of token ids to real text
        """
        return "".join([self.vocab[id] for id in token_ids])


keys = []
values = []


class MHA:
    """
    Multi-head attention for transformer block.
    This is the core functionality of a transformer!
    """

    def __init__(self, wq, wk, wv, wo, heads, head_dim, embed_dim):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.heads = heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim

    def forward(self, input, layer):
        Q = linear(self.wq, input)
        K = linear(self.wk, input)
        V = linear(self.wv, input)

        # K-V cache
        keys.append(K)
        values.append(V)

        concat_scores = []
        for h in range(self.heads):
            # compute attention for each head
            a = h * self.head_dim
            b = a + self.head_dim

            q_h = Q[a:b]
            k_h = [ki[a:b] for ki in keys[layer]]
            v_h = [vi[a:b] for vi in values[layer]]

            # q, k, v are vectors
            # now find qk^T / sqrt(d_k) which is
            scores = [
                sum([q_h[i] * k_h[t][i] for i in range(len(q_h))])
                / Value(math.sqrt(self.head_dim))
                for t in range(len(k_h))
            ]

            weights = softmax(scores)
            attn_scores = [
                sum(weights[i] * v_h[i][t] for i in range(len(weights)))
                for t in range(len(v_h))
            ]
            concat_scores += [attn_scores]

        # merge heads then use the final output layer
        y = linear(self.wo, concat_scores)
        return y


class GPT2:
    """
    Final GPT2 Architecture.
    """

    def __init__(self, state_dict, config, token_emb, pos_emb):
        self.state_dict = state_dict
        self.config = config
        self.token_emb = token_emb
        self.pos_emb = pos_emb

    def forward(self, token_id, pos_id, keys, values):
        token_emb = self.token_emb[token_id]
        pos_emb = self.pos_emb[pos_id]
        embedding = token_emb + pos_emb
        input = rmsnorm(embedding)

        for i in range(self.config.layers):
            residual = input
            mha = MHA(
                self.state_dict[f"layer_{i}_attn_wq"],
                self.state_dict[f"layer_{i}_attn_wk"],
                self.state_dict[f"layer_{i}_attn_wv"],
                self.state_dict[f"layer_{i}_attn_wo"],
                self.config.num_heads,
                self.config.head_dim,
                self.config.embed_dim,
            )
            attn_out = mha.forward(input, i)

            x = [attn_out[i] + residual[i] for i in range(len(attn_out))]
            x_residual = x
            x = rmsnorm(x)

            ## Feed Forward
            l1 = linear(self.state_dict[f"layer_{i}_fc1"], x)
            l1_relu = [x.relu() for x in l1]
            l2 = linear(self.state_dict[f"layer_{i}_fc1"], l1_relu)
            x = l2 + x_residual

            input = x
            output = x

        logits = linear(self.state_dict["output_head"], output)
        return logits
