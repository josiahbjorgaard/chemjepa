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

    def forward(
            self,
            batch,
    ):
        tokens = self.model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask']).last_hidden_state
        return tokens


class PredictorTokenTransform(nn.Module):
    def __init__(self, positional_encoder, padding_token=1):
        """

        :param positional_encoder: generally from encoder.embeddings.position_embeddings
        """
        super().__init__()
        self.transform_mlp = MLP(768, 1, 768, 1, dropout=0.0) #MLP(768, 1, 768, 1)
        self.positional_encoder = positional_encoder #Function to get positional encoding
        self.mask_token = nn.Parameter(torch.randn(768))
        self.padding_token = padding_token

    def encoder(self,
            batch, #Generally 1 for e.g. ChemBERTa
            ):
        """
        Take input_ids/attention mask for mask tokens (where input_ids are mask tokens and attention mask is the location of mask tokens)
        :param x:
        :return: Encoding of the mask tokens with the same attention mask
        """
        mask = batch['input_ids'].ne(self.padding_token).int() #Input ids that aren't the padding token are encoded...
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask #Create the number of the indices
        encs = incremental_indices.long() + self.padding_token #Some number for the position embedding - which is +1 b/c that's how BERTa does it
        encs = self.positional_encoder(encs) #Position embeddings
        tres = batch['transform'].unsqueeze(1).float() #This must be the transform token or signal
        trns = self.transform_mlp(tres).unsqueeze(1).repeat(1, 512, 1)
        return trns + encs #Add the transform signal and the positional embedding...

    def forward(self, xbatch, batch):
        """
        :param x: Context encodings
        :param batch: Unmasked target batch which includes the target_mask and transform values
        :return: batch for predictor which includes the context encodings and the target masked token encodings
        """
        # Encode positions and transform
        ptokens = self.encoder(batch)

        # Now we need to select just the tokens for prediction
        ptokens = [t[a] for t, a in zip(ptokens, batch['target_mask'])]
        max_len = max([t.shape[0] for t in ptokens])
        ptokens = [F.pad(t, [0, 0, 0, max_len - t.shape[0]], value=float('nan')) for t in ptokens]
        ptokens = torch.stack(ptokens) #Reduced size to just the prediction tokens
        pattention_mask = (~ptokens[:, :, 0].isnan()).to(torch.long) #NaN values are to be attention masked
        ptokens = torch.nan_to_num(ptokens, self.padding_token) #Set them to the padding token

        # Add the base token
        ptokens = ptokens + self.mask_token.repeat(ptokens.shape[0], ptokens.shape[1], 1)


        #Now we append them to the batch and return (tokens for predictor, mask for predictor, target tokens mask)
        return torch.cat([xbatch['input_ids'], ptokens], dim=1), \
               torch.cat([xbatch['attention_mask'], pattention_mask], dim=1), \
               torch.cat([xbatch['attention_mask'] * 0, pattention_mask], dim=1).to(torch.bool)


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
            tokens = layer(tokens, padding_mask=padding.to(torch.bool))
        return tokens


class MLP(torch.nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.0, **kwargs ):
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
                 mask_token_predictor = True,
                 **kwargs):
        super().__init__()
        self.run_predictor = run_predictor
        self.mask_token_predictor = mask_token_predictor
        if encoder_config['type'] == 'chemberta':
            self.encoder = HFEncoder(**encoder_config, embedding_config=embedding_config)
        else:
            self.encoder = CJEncoder(**encoder_config, embedding_config=embedding_config)
        self.dim = encoder_config['hidden_size']
        if 'weights' in encoder_config:
            #self.encoder = nn.DataParallel(self.encoder)
            load_model(self.encoder, encoder_config['weights'])
            #self.encoder = self.encoder.module
        self.predictor = CJPredictor(**predictor_config)
        if 'weights' in predictor_config:
            #self.predictor = nn.DataParallel(self.predictor)
            load_model(self.predictor, predictor_config['weights'])
            #self.predictor = self.predictor.module
        if encoder_config['freeze_layers'] > 0:
            print(f"Freezing {encoder_config['freeze_layers']} encoder layers")
            if encoder_config['type'] == 'chemberta':
                modules_to_freeze = [self.encoder.model.embeddings,
                                     #self.encoder.transform_encoder,
                                     self.encoder.model.encoder.layer[:predictor_config['freeze_layers']]]
            else:
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
        if self.mask_token_predictor:
            self.ptform = PredictorTokenTransform(self.encoder.model.embeddings.position_embeddings)
            load_model(self.ptform, predictor_config['mask_weights'])
            module_to_freeze = [self.ptform.transform_mlp,self.ptform.positional_encoder,self.ptform.mask_token]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
        else:
            self.ptform = None
        self.pooling_type = pooling_type
    def unfreeze_layers(self, layers):
        raise NotImplementedError
        #return
    def forward(self, batch):
        #output = self.encoder(batch)
        transform = torch.zeros(batch['input_ids'].shape[0],
                                device=batch['input_ids'].device,
                                dtype=torch.long)
        if self.pooling_type == "first":
            target_mask = torch.ones([batch['input_ids'].shape[0], 1],
                                     device=batch['input_ids'].device,
                                     dtype=torch.long)
        else:  # self.pooling_type == "augmented":
            target_mask = batch['attention_mask']

        output = self.encoder(batch)#, {'transform': transform,
                                             #'target_mask': target_mask})
        if self.run_predictor:
            attention_mask = batch['attention_mask']
            if self.mask_token_predictor: #Misnomer - this means to augmented predictor with mask tokens
                n_init = output.shape[1]
                batch['transform'] = torch.zeros(output.shape[0], device=output.device)
                batch['target_mask'] = batch['attention_mask']
                output, attention_mask, _ = self.ptform({'input_ids': output,
                                                    'attention_mask': attention_mask}, batch)
            output = self.predictor(output,
                                    attention_mask.to(torch.bool)
                                    )
            if self.mask_token_predictor: #Trim to mask tokens for fine tuning
                output = output[:, n_init:, :]
                batch['attention_mask'] = attention_mask[:, n_init:]
                
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
                                            mask_token_predictor = predictor_config['fine_tune_with_mask_tokens']
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


