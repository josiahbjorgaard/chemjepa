import torch
from torch import nn
from torch.nn.functional import pad
from torch import Tensor
from typing import Optional
from einops import repeat
from einops.layers.torch import Rearrange
import functools
import math
from collections import defaultdict
from utils.dataset import BatchDropout

def cum_mul(it):
    return functools.reduce(lambda x, y: x * y, it, 1)


class TokenEncoder(nn.Module):
    """
    Just an nn.embedding wrapper
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = 1.0,
        **kwargs
    ):
        super().__init__()
        self.num_embeddings = num_embeddings #debug
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048,  **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return self.dropout(self.pe[: x.size(1)].repeat(x.size(0), 1, 1))


class SequenceEncoder(nn.Module):
    """
    Basic sequence encoder with sinusoidal positional embedding
    """
    def __init__(self,
                 num_embeddings = 400, #Vocab size
                 embedding_dim = 512, #size of embedding vector
                 padding_idx = 0, #padding (no entry) token
                 dropout = 0.0,
                 max_tokens = 1024,
                 **kwargs
                 ):
        super().__init__()
        self.token_encoder = TokenEncoder(num_embeddings, embedding_dim, padding_idx)
        self.positional_encoder = PositionalEncoder(embedding_dim, dropout, max_tokens)

    def forward(self, batch) -> Tensor:
        x_t = self.token_encoder(batch['tokens'])
        x_p = self.positional_encoder(batch['tokens'])
        x = x_t + x_p
        return x, batch['attention_mask']


class SequenceCollator:
    """
    Sequence collator that also works for dense and sparse tabular data
    For sequence data, input {'index':Tensor}
    TODO: add truncation
    """
    def __init__(self,
                 pad_token=0,
                 pad_len=2048,
                 data_col_name='indices',
                 other_col='data',
                 attn_mask=True,
                 **kwargs
                 ):
        self.pad_token = pad_token
        self.pad_len = pad_len
        self.attn_mask = attn_mask
        self.data_col_name = data_col_name
        self.other_col = other_col

    def __call__(self, data):
        data = {self.data_col_name: [index if index is not None else torch.empty([0]) for index in data[self.data_col_name]]}

        collated_data = {
            self.data_col_name: [pad(index, (0, self.pad_len - index.shape[-1]), mode='constant', value=self.pad_token)
                      for index in data[self.data_col_name]]}
        if self.attn_mask:
            collated_data['attention_mask'] = [(padded_index == self.pad_token).to(torch.long) for padded_index in collated_data[self.data_col_name]]
        if self.other_col in data.keys():
            collated_data[self.other_col] = [pad(index, (0, self.pad_len-index.shape[-1]), mode='constant', value=0.0)
                            for index in data[self.other_col]]
        return {k: torch.stack(v) for k,v in collated_data.items()}