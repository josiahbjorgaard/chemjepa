import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from utils.encoders import SequenceEncoder
from utils.tokenizer import SmilesTokenizer
from utils.smiles import rotate_smiles
from transformers import AutoConfig, AutoModel
from einops import rearrange, repeat, pack, unpack
#from collections import defaultdict
from safetensors.torch import load_model

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
                 transform=False,
                 vocab_file='../data/vocab.txt',
                 max_length=128,
                 stop_token=13,
                 mask_token=14,
                 ):
        super().__init__()
        self.mask_size = num_mask
        self.mask_token = mask_token
        self.stop_token = stop_token
        self.transform = transform
        self.max_length=max_length
        if transform:
            self.tokenizer = SmilesTokenizer(vocab_file)
            self.tokenize = lambda x: self.tokenizer(x, max_length=max_length, padding="max_length", return_tensors='pt', truncation=True)

    def forward(self, batch):
        smiles = batch['SMILES']
        device = batch['input_ids'].device
        batch = self.tokenize(smiles)

        if self.transform:
            #Rotation, etc. goes here TBD. Encode via a token to describe the rotation
            vocab_len = len(self.tokenizer.vocab)
            rand_rotate = torch.randint(0,self.max_length, (len(smiles),))
            xsmiles = [rotate_smiles(smile, rot) for smile,rot in zip(smiles, rand_rotate)]
            xbatch = self.tokenize(xsmiles) #batch['input_ids'], batch['attention_mask']
            marker_tokens = (xbatch['input_ids'] == self.stop_token).nonzero(as_tuple=True)
            #Add embedding for rotation/permutation by adding an embedding for each rotation
            for idx,rot in zip(marker_tokens,rand_rotate):
                xbatch['input_ids'][idx] = rot+vocab_len
        else:
            xbatch=batch

        #Masking tokens
        token_counts = xbatch['attention_mask'].sum(dim=1)+1
        #Probably a faster way of doing this
        ntok = xbatch['input_ids'].shape[1]
        xmask = torch.stack([
            torch.zeros(ntok, device=xbatch['input_ids'].device).index_fill_(0,
                                        torch.randperm(c, device=xbatch['input_ids'].device)[:self.mask_size],
                                        1)
                     for c in token_counts]).to(torch.bool)
        xbatch['input_ids'][xmask]=self.mask_token
        xbatch['attention_mask'][xmask]=0
        return dict(batch), dict(xbatch), xmask


class MLP(torch.nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, **kwargs ):
        super().__init__()
        self.model_type = "MLP"
        self.dropout = nn.Dropout(dropout)
        layers = [
                    nn.Linear(ninp, nhid),
                    nn.ReLU()
                 ]
        for n in range(nlayers):
            layers = layers + [
                                    nn.Dropout(dropout),
                                    nn.Linear(nhid, nhid),
                                    nn.ReLU()
                              ]
        layers = layers + [
                                nn.Dropout(dropout),
                                nn.Linear(nhid, ntoken)
                          ]
        self.model = nn.Sequential(*layers)
        print(self.model)
    def forward(self, batch):
        return self.model(batch).squeeze() # #d.squeeze()


class PretrainedHFEncoder(nn.Module):
    def __init__(self, model_path, freeze_layers = 5, pooling_type="None", load_weights = True, **kwargs):
        super().__init__()
        self.model_type = "PretrainedEncoder"
        if load_weights:
            model_config = AutoConfig.from_pretrained(model_path)
            #for k,v in kwargs.items():
            #    model_config['k'] = v
            self.model = AutoModel.from_pretrained(model_path)
        else:
            print(f"WARNING: Not loading model weights")
            model_config = AutoConfig.from_pretrained(model_path)
            #for k,v in kwargs.items():
            #    model_config['k'] = v
            self.model = AutoModel.from_config(model_config)
        if freeze_layers > 0:
            print(f"Freezing {freeze_layers} layers")
            modules_to_freeze = [self.model.embeddings,
                                    self.model.encoder.layer[:freeze_layers]]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
        self.pooling_type = pooling_type
    def unfreeze_layers(self, layers):
        raise NotImplementedError
        #return
    def forward(self, **kwargs):
        output = self.model(**kwargs)
        output = output.last_hidden_state
        if self.pooling_type == "mean":
            embeddings = output
            padding_mask = kwargs['attention_mask']
            padding_mask = repeat(padding_mask, 'b t -> b t e', e=embeddings.shape[-1])
            output = (embeddings * padding_mask).mean(dim=1).squeeze()
        elif self.pooling_type == "first":
            embeddings = output
            output = embeddings[:,0,:]
        return output


class PretrainedCJEncoder(nn.Module):
    def __init__(self, run_predictor, encoder_config, predictor_config, encoder_freeze = 0, predictor_freeze = 0, pooling_type="First", load_weights = True, **kwargs):
        super().__init__()
        self.run_predictor=run_predictor
        self.encoder = CJEncoder(**encoder_config)
        load_model(self.encoder,encoder_config['weights'])
        self.predictor = CJPredictor(**predictor_config)
        load_model(self.predictor,predictor_config['weights'])

        if encoder_freeze > 0:
            print(f"Freezing {freeze_layers} encoder layers")
            modules_to_freeze = [self.model.encoder,
                                    self.model.layers[:freeze_layers]]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
        if predictor_freeze > 0:
            print(f"Freezing {freeze_layers} predictor layers")
            modules_to_freeze = [self.model.layers[:freeze_layers]]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        self.pooling_type = pooling_type
    def unfreeze_layers(self, layers):
        raise NotImplementedError
        #return
    def forward(self, **kwargs):
        output = self.encoder(**kwargs)
        if self.run_predictor
            output = self.predictor(output, kwargs['attention_mask'])
        if self.pooling_type == "mean":
            embeddings = output
            padding_mask = kwargs['attention_mask']
            padding_mask = repeat(padding_mask, 'b t -> b t e', e=embeddings.shape[-1])
            output = (embeddings * padding_mask).mean(dim=1).squeeze()
        elif self.pooling_type == "first":
            embeddings = output
            output = embeddings[:,0,:]
        return output

class FineTuneModel(nn.Module):
    def __init__(self, run_predictor, encoder_config, decoder_config, loss_config):
        super().__init__()
        self.backbone = PretrainedCJEncoder(run_predictor,
                                            encoder_config,
                                            predictor_config,
                                            encoder_freeze=0,
                                            predictor_freeze=0,
                                            pooling_type="First"
                                            )

        if decoder_config.type == "MLP":
            self.decoder_type = "MLP"
            self.decoder = MLP(**decoder_config)
        elif decoder_config.type == "Linear":
            self.decoder_type = "Linear"
            self.decoder = nn.Linear(in_features=decoder_config['ninp'],
                                     out_features=decoder_config['ntoken'])

        if chem_config.no_loss:
            self.loss_fct = None
        else:
            if loss_config.type == "mse":
                self.loss_type = "mse"
                self.loss_fct = nn.MSELoss()
            elif loss_config.type == "bce":
                self.loss_type = "bce"
                self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([loss_config.pos_weight]))
            else:
                self.loss_fct = None

    def forward(
            self,
            labels,
            batch,
            **kwargs,
    ):

        embedding = self.backbone(**batch)
        if self.loss_fct:
            logits = self.decoder(embedding).squeeze()

            loss = self.loss_fct(logits, labels.float())

            return loss, F.sigmoid(logits)
        else:
            raise Exception("remove me to get chem embeddings only")
            return embedding
