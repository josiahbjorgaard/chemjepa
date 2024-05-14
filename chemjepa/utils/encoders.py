import torch
from torch import nn
from torch.nn.functional import pad
from torch import Tensor
from typing import Optional
from utils.tokenizer import SmilesTokenizer
from utils.smiles import rotate_smiles, SmilesTransformations
import functools
import math
from collections import defaultdict
from transformers import AutoTokenizer
from functools import partial

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
        if (x >= self.num_embeddings).sum():
            raise Exception(f"found {x.max()}")
        if (~torch.isfinite(x)).sum():
            raise Exception("Found non finite vbalue")
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
                 padding_token = 0, #padding (no entry) token
                 dropout = 0.0,
                 max_tokens = 1024,
                 **kwargs
                 ):
        super().__init__()
        self.token_encoder = TokenEncoder(num_embeddings, embedding_dim, padding_token)
        self.positional_encoder = PositionalEncoder(embedding_dim, dropout, max_tokens)

    def forward(self, batch) -> Tensor:
        x_t = self.token_encoder(batch['input_ids'])
        x_p = self.positional_encoder(batch['input_ids'])
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


class CJPreprocessCollator:
    def __init__(self, 
                 num_mask=4,
                 transform=None,
                 mask=True,
                 rotate = "all",
                 vocab_file='../data/vocab.txt',
                 max_length=128,
                 stop_token=13,
                 mask_token=14,
                 smiles_col="SMILES",
                 encoder = ""
                 ):
        super().__init__()
        self.smiles_col=smiles_col
        self.mask_size = num_mask
        self.mask_token = mask_token
        self.stop_token = stop_token
        self.transform = transform
        self.rotate = rotate
        self.mask = mask
        self.max_length = max_length
        self.smiles_transform = SmilesTransformations(mask_size=num_mask, transform=transform)
        self.encoder = encoder
        if self.encoder != 'chemberta':
            self.tokenizer = SmilesTokenizer(vocab_file)
            self.tokenize = lambda x: self.tokenizer(x, max_length=max_length, padding="max_length", return_tensors='pt', truncation=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", max_len=512, return_tensors='pt')
            self.tokenize = partial(self.tokenizer, padding = 'max_length', truncation=True, return_tensors='pt')
            self.mask_token = 1 #Override mask token

    def __call__(self, batch):
        #print(batch)
        smiles = [x[self.smiles_col] for x in batch]
        metadata = {k:[x[k] for x in batch] for k in batch[0].keys()}
        if self.transform == "old":
            vocab_len = len(self.tokenizer.vocab)
            #First rotate to a context state. This one gets masked.
            rand_rotate = torch.randint(0, self.max_length, (len(smiles),))
            xsmiles = [rotate_smiles(smile, rot) for smile, rot in zip(smiles, rand_rotate)]
            rand_rotate = [rot if smi else 0 for smi, rot in zip(xsmiles, rand_rotate)]
            xsmiles = [xsmi if xsmi else smi for xsmi, smi in zip(xsmiles, smiles)]
            xbatch = self.tokenize(xsmiles) #batch['input_ids'], batch['attention_mask']
            #marker_tokens = (xbatch['input_ids'] == self.stop_token).nonzero(as_tuple=True)
            #No token for this
            #for idx, rot in zip(marker_tokens,rand_rotate):
            #    xbatch['input_ids'][idx] = vocab_len+1

            #Rotate from the prediction state to get a target state
            rand_rotate = torch.randint(0,self.max_length, (len(xsmiles),))
            nsmiles = [rotate_smiles(xsmile, rot) for xsmile, rot in zip(xsmiles, rand_rotate)]
            smiles = [n if n else s for s,n in zip(xsmiles,nsmiles)]
            batch = self.tokenize(smiles) #batch['input_ids'], batch['attention_mask']
            #marker_tokens = (batch['input_ids'] == self.stop_token).nonzero(as_tuple=True)

            #Add info for transformation
            batch['transform'] = rand_rotate
        elif self.transform:
            #The new transform for smiles with matching mask tokens in transformations
            #N.B. the token for mask is 256 = '*' in this transformation
            if self.rotate == "all":
                rand_rotate_init = torch.randint(0, self.max_length, (len(smiles),))
                rand_rotate = torch.randint(0, self.max_length, (len(smiles),))
            elif self.rotate == "init":
                rand_rotate_init = torch.randint(0, self.max_length, (len(smiles),))
                rand_rotate = torch.zeros_like(rand_rotate_init)
            else:
                rand_rotate_init = torch.zeros(len(smiles))
                rand_rotate = torch.zeros(len(smiles))
            xsmiles, rsmiles, mrsmiles, res_rotate = [], [], [], []
            for s, r, ri in zip(smiles, rand_rotate, rand_rotate_init):
                #initially rotated, initially rotated and masked,
                #further rotated, further rotated with mask
                _, xms, ts, tms, rot = self.smiles_transform(s, int(r), int(ri))
                xsmiles.append(xms) #Masked context smiles
                rsmiles.append(ts) #Unmasked target smiles
                mrsmiles.append(tms) #Masked target smiles
                res_rotate.append(rot)
            if self.encoder == 'chemberta':
                xsmiles = [s.replace('*','<mask>') for s in xsmiles]
                rsmiles = [s.replace('*','<mask>') for s in rsmiles]
                mrsmiles = [s.replace('*','<mask>') for s in mrsmiles]
            xbatch = self.tokenize(xsmiles) #For context encoder
            batch = self.tokenize(rsmiles) # For target encoder
            mbatch = self.tokenize(mrsmiles) #For mask for predictor
            if self.encoder == 'chemberta':
                pxmask = mbatch['input_ids'] == 4 #Get mask tokens
                xmask = xbatch['input_ids'] == 4
                #mbatch['input_ids'][pxmask] = 1 #Change to padding token for encoder
                #xbatch['input_ids'][xmask] = 1
                batch['transform'] = torch.LongTensor(res_rotate)
            else:
                pxmask = mbatch['input_ids'] == 256 #Mask for predictor
                xmask = xbatch['input_ids'] == 256 #Mask for context encoder
                vocab_len = len(self.tokenizer.vocab)
                batch['transform'] = torch.LongTensor(res_rotate) + vocab_len
            #For new mask encodings, we need to set up the target mask positional
            # encodings + embeddings later. To do that we can use the pxmask
            # in the batch (target encoding) data
            batch['target_mask'] = pxmask
        else:
            batch = self.tokenize(smiles)
            batch['transform'] = None
            xbatch=batch
        if self.mask:
            ## Start old junk
            if not 'xmask' in locals():
                #Masking tokens
                token_counts = xbatch['attention_mask'].sum(dim=1)
                #Probably a faster way of doing this
                ntok = xbatch['input_ids'].shape[1]
                xmask = torch.stack([
                    torch.zeros(ntok, device=xbatch['input_ids'].device).index_fill_(0,
                                                torch.randperm(c, device=xbatch['input_ids'].device)[:self.mask_size],
                                                1)
                         for c in token_counts]).to(torch.bool)
            if self.transform == "old":
                for idx, (ixmask, irot) in enumerate(zip(xmask, rand_rotate)):
                    xbatch['input_ids'][idx, ixmask] = self.mask_token + irot
            ## End old junk
            else:
                xbatch['input_ids'][xmask] = self.mask_token
            xbatch['attention_mask'][xmask] = 0
        else:
            return dict(batch)
        return dict(batch), dict(xbatch), xmask, metadata
