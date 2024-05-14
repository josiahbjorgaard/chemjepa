import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from utils.encoders import SequenceEncoder
from transformers import AutoConfig, AutoModel
from einops import rearrange, repeat, pack, unpack
#from collections import defaultdict
from safetensors.torch import load_model
#from transformers import AutoTokenizer
#from functools import partial

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def exists(val):
    return val is not None

def make_predictor_tokens(encoder, transform, target_mask):
    target_token_batch = {'input_ids': transform.unsqueeze(1).repeat(1, target_mask.shape[1]),
                          'attention_mask': target_mask}
    ptokens, pattention_mask = encoder(target_token_batch)
    #Now we need to select just the tokens for prediction and append them
    ptokens = [t[a] for t, a in zip(ptokens, pattention_mask)]
    max_len = max([t.shape[0] for t in ptokens])
    ptokens = [F.pad(t, [0, 0, 0, max_len-t.shape[0]], value=float('nan')) for t in ptokens]
    ptokens = torch.stack(ptokens)
    pattention_mask = (~ptokens[:,:,0].isnan()).to(torch.long)
    ptokens = torch.nan_to_num(ptokens, 0.0)
    return ptokens, pattention_mask

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
        self.norm = torch.nn.LayerNorm(dim)         
    
    def forward(self, batch, attn_mask=None, padding_mask=None):
        if (~torch.isfinite(batch)).sum():
            raise Exception("nan in batch")
        batch = self.norm(batch)
        batch = self.attn(batch, attn_mask=attn_mask, key_padding_mask = padding_mask) + batch
        if (~torch.isfinite(batch)).sum():
            raise Exception("nan in batch after attn")
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
            mask = None
    ):
        tokens, attention_mask = self.encoder(batch)
        if mask is not None:
            masked_tokens = tokens[mask] #Skip the masked tokens
        padding = attention_mask.to(torch.bool)
        for idx, layer in enumerate(self.layers):
            tokens = layer(tokens, padding_mask=padding)
        if mask is not None:
            tokens[mask] = masked_tokens
        return tokens


class HFEncoder(nn.Module):
    def __init__(
            self,
            load_weights = True,
            model_path = "seyonec/PubChem10M_SMILES_BPE_450k",
            freeze_layers = 6,
            **kwargs
            ):
        super().__init__()
        self.model_type = "PretrainedEncoder"
        print(f"Got kwargs: {kwargs}")
        
        #tokenizer = AutoTokenizer.from_pretrained(hf_path, max_len=512)
        #self.encoder = partial(tokenizer, padding = 'max_length', truncation=True)

        if load_weights:
            model_config = AutoConfig.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
        else:
            print(f"WARNING: Not loading model weights")
            model_config = AutoConfig.from_pretrained(model_path)
            self.model = AutoModel.from_config(model_config)
        if freeze_layers > 0:
            print(f"Freezing {freeze_layers} layers")
            modules_to_freeze = [self.model.embeddings,
                                    self.model.encoder.layer[:freeze_layers]]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
        print(self.model)
        self.transform_encoder =  MLP(768, 1, 768, 1)
        
    def encoder(self,
            x):
        padding_idx = 1
        x['input_ids'][:,0]=1
        x['input_ids'][:,-1]=1
        mask = x['input_ids'].ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        encs =  incremental_indices.long() + padding_idx
        encs = self.model.embeddings.position_embeddings(encs)
        tres = x['input_ids'][:,0].unsqueeze(1).float()
        trns = self.transform_encoder(tres).unsqueeze(1).repeat(1,512,1)
        return trns + encs, x['attention_mask']
    
    def forward(
            self,
            batch,
            #mask = None
    ):
        #tokens = self.encoder(batch)
        #input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        #if mask is not None:
        #    masked_tokens = tokens[mask] #Skip the masked tokens
        tokens = self.model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask']).last_hidden_state
        #if mask is not None:
        #    tokens[mask] = masked_tokens
        return tokens


class CJPredictor(nn.Module):
    def __init__(
            self,
            hidden_size,
            layers,
            dim_head=64,
            heads=8,
            ff_mult=4,
            transform=False,
            **kwargs
    ):
        super().__init__()
        print(f"Got kwargs: {kwargs}")

        self.heads = heads
        # transformer
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(TransformerLayer(hidden_size, dim_head, heads, ff_mult))
        self.transform_mix = MLP(hidden_size, hidden_size+1, hidden_size, 1) if transform == "mix" else None
    def forward(
            self,
            tokens,
            padding,
            mask=None,
            transform=None,
    ):
        """
        mask is 2d (batch, token inde)
        padding is 2d (batch, token index)
        transform is 1d (batch)
        tokens is 3d (batch, token, embedding)
        """

        #Doing below with torch scatter would probably be faster
        if mask is not None and transform is not None and self.transform_mix is not None:
            mask_tokens = torch.stack([torch.cat([tokens[idx[0],idx[1],:].squeeze(),
                                                  transform[idx[0]].unsqueeze(0)])
                                       for idx in mask.nonzero()])
            transformed_tokens = self.transform_mix(mask_tokens)
            for i, idx in enumerate(mask.nonzero()):
                tokens[idx[0], idx[1], :] = transformed_tokens[i, :]

        for idx, layer in enumerate(self.layers):
            tokens = layer(tokens, padding_mask=padding)
        return tokens


class MLP(torch.nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.1, **kwargs ):
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


class AttentivePooling(torch.nn.Module):
    def __init__(self, ntoken, nhid, dropout=0.1, dim_head=32, heads=8, **kwargs ):
        super().__init__()
        self.model_type = "AttentivePooling"
        self.dropout = nn.Dropout(dropout)
        self.return_tokens = nn.Parameter(torch.randn(ntoken, nhid))
        self.attn_pool = Attention(dim=nhid, dim_head=dim_head, heads=heads)
        self.linear = nn.Linear(in_features=nhid, out_features=1)
    def forward(self, tokens, attention_mask):

        pooled_tokens = self.attn_pool(self.return_tokens.unsqueeze(0).repeat(tokens.shape[0],1,1), tokens,
                                       key_padding_mask=attention_mask) + self.return_tokens
        output = self.linear(pooled_tokens)
        return output


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
    def __init__(self, run_predictor,
                 embedding_config,
                 encoder_config,
                 predictor_config,
                 pooling_type="first",
                 class_token_predictor = False,
                 **kwargs):
        super().__init__()
        self.run_predictor = run_predictor
        self.class_token_predictor = class_token_predictor
        self.encoder = CJEncoder(embedding_config,**encoder_config)
        self.dim = encoder_config['hidden_size']
        if 'weights' in encoder_config:
            load_model(self.encoder, encoder_config['weights'])
        self.predictor = CJPredictor(**predictor_config)
        if 'weights' in predictor_config:
            load_model(self.predictor, predictor_config['weights'])

        if encoder_config['freeze_layers'] > 0:
            print(f"Freezing {encoder_config['freeze_layers']} encoder layers")
            modules_to_freeze = [self.encoder.encoder,
                                    self.encoder.layers[:encoder_config['freeze_layers']]]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
        if predictor_config['freeze_layers'] > 0:
            print(f"Freezing {predictor_config['freeze_layers']} predictor layers")
            modules_to_freeze = [self.predictor.layers[:predictor_config['freeze_layers']]]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        self.pooling_type = pooling_type
    def unfreeze_layers(self, layers):
        raise NotImplementedError
        #return
    def forward(self, batch):
        output = self.encoder(batch)
        if self.run_predictor:
            if self.class_token_predictor: #Misnomer - this means to augmented predictor with mask tokens
                #assert self.pooling_type == 'first'
                #assert 'transform' in batch.keys() and 'target_mask' in batch.keys()
                n_init = batch['input_ids'].shape[1]
                transform = torch.zeros(batch['input_ids'].shape[0],
                        device=batch['input_ids'].device,
                        dtype=torch.long)
                if self.pooling_type == "first":
                    target_mask = torch.ones([batch['input_ids'].shape[0],1], 
                            device=batch['input_ids'].device,
                            dtype=torch.long)
                else: # self.pooling_type == "augmented":
                    target_mask = batch['attention_mask']
                
                x, a = make_predictor_tokens(self.encoder.encoder,
                                             transform,
                                             target_mask,
                                             )
                output = torch.cat([x, output], dim=1)
                batch['attention_mask'] = torch.cat([a, batch['attention_mask']], dim=1)
            output = self.predictor(output,
                                    batch['attention_mask'].to(torch.bool)
                                    )
            if self.class_token_predictor: #Trim to mask tokens for fine tuning
                output = output[:, n_init:, :]
                batch['attention_mask'] = batch['attention_mask'][:, n_init:]

        if self.pooling_type == "mean":
            embeddings = output
            padding_mask = batch['attention_mask']
            padding_mask = repeat(padding_mask, 'b t -> b t e', e=embeddings.shape[-1])
            output = (embeddings * padding_mask).mean(dim=1).squeeze()
        elif self.pooling_type == "first":
            embeddings = output
            output = embeddings[:, 0, :]
        return output, batch['attention_mask']


class FineTuneModel(nn.Module):
    def __init__(self, run_predictor,
                 embedding_config,
                 encoder_config,
                 predictor_config,
                 decoder_config,
                 loss_config,
                 ):
        super().__init__()
        self.backbone = PretrainedCJEncoder(run_predictor,
                                            embedding_config,
                                            encoder_config,
                                            predictor_config,
                                            pooling_type=decoder_config.pooling_type,
                                            class_token_predictor = predictor_config['fine_tune_with_class_token']
                                            )

        if decoder_config.type == "MLP":
            self.decoder_type = "MLP"
            self.decoder = MLP(**decoder_config)
        elif decoder_config.type == "Linear":
            self.decoder_type = "Linear"
            self.decoder = nn.Linear(in_features=decoder_config['ninp'],
                                     out_features=decoder_config['ntoken'])
        elif decoder_config.type == "Attentive":
            self.decoder_type = "Attentive"
            self.decoder = AttentivePooling(**decoder_config)

        if loss_config.type == "mse":
            self.loss_type = "mse"
            self.loss_fct = nn.MSELoss()
        elif loss_config.type == "mae":
            self.loss_type = 'mae'
            self.loss_fct = nn.L1Loss()
        elif loss_config.type == "bce":
            self.loss_type = "bce"
            self.loss_fct = nn.BCEWithLogitsLoss()#pos_weight=torch.Tensor([loss_config.pos_weight]))
        else:
            self.loss_fct = None

    def forward(
            self,
            labels,
            batch,
            **kwargs,
    ):
        batch['attention_mask'] = batch['attention_mask'].to(torch.bool)
        embedding, attn_mask = self.backbone(batch)
        if self.loss_fct:
            if self.decoder_type == "Attentive":
                logits = self.decoder(embedding, attn_mask.to(torch.bool)).squeeze()
            else:
                logits = self.decoder(embedding).squeeze()
            loss = self.loss_fct(logits, labels)
            if self.loss_type == 'bce':
                return loss, F.sigmoid(logits)
            else:
                return loss, logits
        else:
            raise Exception("remove me to get chem embeddings only")
            return embedding


