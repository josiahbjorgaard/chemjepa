import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from utils.encoders import SequenceEncoder
from utils.tokenizer import SmilesTokenizer
from utils.smiles import rotate_smiles

from einops import rearrange, repeat, pack, unpack

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def exists(val):
    return val is not None

# bias-less layernorm
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# geglu feedforward
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class FeedForward(nn.Module):
    def __init__(self,
                dim, 
                mult=4):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)
    
        self.feedforward = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            nn.Linear(inner_dim, dim, bias=False)
            )
    def forward(self, batch):
        return self.feedforward(batch)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
            self,
            x,
            context=None,
            attn_mask=None,
            key_padding_mask=None,
            return_attn=False
    ):
        kv_x = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(attn_mask):
            sim = sim.masked_fill(attn_mask, -torch.finfo(sim.dtype).max)
        if exists(key_padding_mask):
            key_padding_mask = repeat(key_padding_mask, "b i -> b h j i", h=self.heads, j=sim.shape[-2])
            torch.save(key_padding_mask,'key_padding_mask.pt')
            sim = sim.masked_fill(key_padding_mask, -torch.finfo(sim.dtype).max)
        
        attn = sim.softmax(dim=-1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v) #j is number of toks
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        if return_attn:
            return self.to_out(out), attn
        else:
            return self.to_out(out)

# attention
class TransformerLayer(nn.Module):
    def __init__(self, dim, dim_head, heads, ff_mult):
        super().__init__()
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.num_heads = heads
        self.ff = FeedForward(dim=dim, mult=ff_mult)
        self.norm = LayerNorm(dim)         
    
    def forward(self, batch, attn_mask=None, padding_mask=None):
        batch = self.norm(batch)
        batch = self.attn(batch, attn_mask=attn_mask, key_padding_mask = padding_mask) + batch
        batch = self.norm(batch)
        batch = self.ff(batch) + batch
        return batch



class CJLoss(nn.Module):
    """
    """
    def __init__(
            self,
    ):
        super().__init__()

    def forward(
            self,
    ):
        return


class CJEncoder(nn.Module):
    def __init__(
            self,
            embedding_config,
            hidden_size,
            layers,
            dim_head=64,
            heads=8,
            ff_mult=4,
            **kwargs
    ):
        super().__init__()
        print(f"Got kwargs: {kwargs}")

        self.encoder = SequenceEncoder(**embedding_config) #Contains nn.Embeddings

        # transformer
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(TransformerLayer(hidden_size, dim_head, heads, ff_mult))

    def forward(
            self,
            batch,
    ):
        tokens, attention_mask = self.encoder(batch)
        padding = attention_mask.to(torch.bool)
        for idx, layer in enumerate(self.layers):
            tokens = layer(tokens, padding_mask=padding)
        return tokens

class CJPredictor(nn.Module):
    def __init__(
            self,
            hidden_size,
            layers,
            dim_head=64,
            heads=8,
            ff_mult=4,
            **kwargs
    ):
        super().__init__()
        print(f"Got kwargs: {kwargs}")

        self.heads = heads
        # transformer
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(TransformerLayer(hidden_size, dim_head, heads, ff_mult))

    def forward(
            self,
            tokens,
            padding,
    ):
        for idx, layer in enumerate(self.layers):
            tokens = layer(tokens, padding_mask=padding)
        return tokens


class CJPreprocess(nn.Module):
    def __init__(self, num_mask=4,
                 mask_token=14,
                 transform=False,
                 vocab_file='../data/vocab.txt',
                 max_length=128
                 stop_token=13):
        super().__init__()
        self.mask_size = num_mask
        self.mask_token = mask_token
        self.transform = transform
        self.max_length=max_length
        if transform:
            self.tokenizer = SmilesTokenizer(vocab_file)
            self.tokenize = lambda x: self.tokenizer(x, max_length=max_length, padding="max_length")

    def forward(self, batch):
        smiles = batch['SMILES']
        tokens, attention_mask = self.tokenize(smiles) #batch['input_ids'], batch['attention_mask']

        if transform:
            #Rotation, etc. goes here TBD. Encode via a token to describe the rotation
            vocab_len = len(self.tokeizer.vocab)
            rand_rotate = torch.randint(0,self.max_length, len(smiles))
            xsmiles = [rotate_smiles(smile, rot) for smile in zip(smiles, rot)]
            xtokens, xattention_mask = self.tokenize(xsmiles) #batch['input_ids'], batch['attention_mask']
            marker_tokens = (xtokens == self.stop_token).nonzero(as_tuple=True)
            print(marker_tokens)
            #Add embedding for rotation/permutation by adding an embedding for each rotation
            for idx,rot in zip(marker_tokens,rand_rotate):
                xtokens[idx] = rot+vocab_len
        else:
            xtokens, xattention_mask = tokens, attention_mask

        #Masking tokens
        token_counts = xattention_mask.sum(dim=1)+1
        #Probably a faster way of doing this
        ntok = xtokens.shape[1]
        xmask = torch.stack([
            torch.zeros(ntok, device=tokens.device).index_fill_(0,
                                        torch.randperm(c, device=xtokens.device)[:self.mask_size],
                                        1)
                     for c in token_counts]).to(torch.bool)
        xbatch = batch
        xbatch['input_ids'] = xtokens
        xbatch['attention_mask'] = xattention_mask
        xbatch['input_ids'][xmask]=self.mask_token
        xbatch['attention_mask'][xmask]=0
        return batch, xbatch, xmask