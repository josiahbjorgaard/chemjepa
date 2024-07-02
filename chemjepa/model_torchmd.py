import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from einops import repeat, pack, rearrange
from collections import defaultdict

def chemcpa_collator(batch, smiles_col='SMILES'):
    rebatch={'smiles':[],
             'labels':[],
             'chem_batch':defaultdict(list),}
    if 'sample_index' in batch[0].keys():
        rebatch['sample_index'] = []
    for i in batch:
        for k,v in i.items():
            if smiles_col in k:
                rebatch['smiles'].append(i[k])
            elif 'chem' in k:
                rebatch['chem_batch'][k[5:]].append(i[k])
            elif 'labels' in k:
                rebatch['labels'].append(i[k])
            elif 'sample_index' in k:
                rebatch['sample_index'].append(i[k])
    #rebatch['smiles'] = torch.stack(rebatch['smiles'])
    rebatch['chem_batch'] = {k: torch.stack(v) for k,v in rebatch['chem_batch'].items()}
    rebatch['labels'] = torch.stack(rebatch['labels'])
    if 'sample_index' in rebatch.keys():
        rebatch['sample_index'] = torch.stack(rebatch['sample_index'])
    #assert len(rebatch['smiles']) == len(rebatch['chem_batch']) f"{len(rebatch['smiles']) // {len(rebatch['chem_batch'])}"
    return rebatch

def chem_collator(batch):
    rebatch={'chem_batch':defaultdict(list),
             'labels':[],}
    if 'sample_index' in batch[0].keys():
        rebatch['sample_index'] = []
    for i in batch:
        for k,v in i.items():
            if 'chem' in k:
                rebatch['chem_batch'][k[5:]].append(i[k])
            elif 'labels' in k:
                rebatch['labels'].append(i[k])
            elif 'sample_index' in k:
                rebatch['sample_index'].append(i[k])
    rebatch['chem_batch'] = {k: torch.stack(v) for k,v in rebatch['chem_batch'].items()}
    rebatch['labels'] = torch.stack(rebatch['labels'])
    if 'sample_index' in rebatch.keys():
        rebatch['sample_index'] = torch.stack(rebatch['sample_index'])
    return rebatch

def collator(batch):
   # print(batch)
    rebatch={'chem_batch':defaultdict(list),
             'cell_batch':defaultdict(list),
             'labels':[],}
    if 'dose_um' in batch[0].keys():
        rebatch['dosage']=[]
    if 'variance' in batch[0].keys():
        rebatch['variance']=[]
    if 'gene_batch' in batch[0].keys():
        rebatch['gene_batch'] = defaultdict(list)
    if 'sample_index' in batch[0].keys():
        rebatch['sample_index'] = []
    for i in batch:
        for k,v in i.items():
            if 'chem' in k:
                rebatch['chem_batch'][k[5:]].append(i[k])
            elif 'cell' in k and 'length' not in k:
                rebatch['cell_batch'][k[5:]].append(i[k])
            elif 'labels' in k:
                rebatch["labels"].append(i[k])
            elif 'dose_um' in k:
                rebatch['dosage'].append(i[k])
            elif 'variance' in k:
                i[k] = i[k].nan_to_num(20**2)
                rebatch["variance"].append(i[k])
            elif 'sample_index' in k:
                rebatch['sample_index'].append(i[k])
    rebatch['chem_batch'] = {k: torch.stack(v) for k,v in rebatch['chem_batch'].items()}
    rebatch['cell_batch'] = {k: torch.stack(v) for k,v in rebatch['cell_batch'].items()}
    rebatch['labels'] = torch.stack(rebatch['labels'])
    if 'variance' in rebatch.keys():
        rebatch['variance'] = torch.stack(rebatch['variance'])
    if 'dosage' in rebatch.keys():
        rebatch['dosage'] = torch.stack(rebatch['dosage'])
    if 'sample_index' in rebatch.keys():
        rebatch['sample_index'] = torch.stack(rebatch['sample_index'])
    return rebatch

class PerturbationLoss(nn.Module):
    def __init__(self, importance_weighting = "z-score", **kwargs):
        super().__init__()
        self.importance_weighting = importance_weighting

    def compute_weights(self, variance, label):
        eps = 1e-1
        if self.importance_weighting == "identity":
            return 1
        elif self.importance_weighting == "equi-z-score":
            return torch.abs(label)+eps
        elif self.importance_weighting == "z-score":
            # z-score assuming logfc has 0 mean
            assert label
            return (torch.abs(label) + eps) / (torch.sqrt(variance) + eps)
        elif self.importance_weighting == "variance":
            return 1 / (variance + eps)
        elif self.importance_weighting == "standard deviation":
            return 1 / (torch.sqrt(variance) + eps)
        raise ValueError(f"The input {self.importance_weighting} was not set correctly")

    def weighted_mse_loss(self, inputs, labels, weights):
        return (weights * (inputs - labels) ** 2).sum() / weights.sum()

    def forward(self, inputs, labels, variance):
        weights = self.compute_weights(variance=variance, label=labels)
        assert weights is not None
        loss = self.weighted_mse_loss(inputs, labels, weights)
        return loss


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


class TransformerEncoderModel(nn.Transformer):
    """Container module with an encoder.py, a recurrent or transformer module, and a decoder.
    This is a process/decode model"""
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, **kwargs):
        super(TransformerEncoderModel, self).__init__(d_model=ninp,
                                               nhead=nhead,
                                               dim_feedforward=nhid,
                                               num_encoder_layers=nlayers,
                                               batch_first = True) #Input is [N,S,E] else [S,N,E]
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        #nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, padding_mask):
        output = self.encoder(src, src_key_padding_mask=padding_mask)# src [N,S,E], padding [N,S]
        output = self.decoder(output)
        return output #F.log_softmax(output, dim=-1)


class TransformerDecoderModel(nn.Module):
    """Container module with a cross-attending decoder."""
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, pooling_type='mean', dropout=0.1, **kwargs):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=ninp,
                                                   nhead=nhead,
                                                   dim_feedforward=nhid,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        #self.dropout = nn.Dropout(dropout)
        self.pooling_type = pooling_type
        if pooling_type == "cross-attention":
            multihead_attn = nn.MultiheadAttention(ninp, nhead, batch_first=True, dropout=dropout)
            linear = nn.Linear(ninp, ntoken)
            self.output = nn.Sequential(multihead_attn, linear)
        elif pooling_type == "mean-token":
            self.ntoken = ntoken
            self.output = nn.Linear(ninp, 1)
        else:
            self.output = nn.Linear(ninp, ntoken)
        
        #self.model = nn.Sequential(decoder, dropout, output)

    def forward(self, sequence_a, sequence_b, padding_mask_a=None, padding_mask_b=None, output_queries=None):
        #print(self.model)
        last_hidden_state = self.decoder(
                                tgt=sequence_a,
                                memory=sequence_b,
                                tgt_key_padding_mask=padding_mask_a.to(torch.bool),
                                memory_key_padding_mask=padding_mask_b.to(torch.bool)
                                )
        
        if self.pooling_type == "mean":
            embeddings = last_hidden_state
            padding_mask = padding_mask_a
            padding_mask = repeat(padding_mask, 'b t -> b t e', e=embeddings.shape[-1])
            output = (embeddings * padding_mask).mean(dim=1).squeeze()
            output = self.output(output)
        elif self.pooling_type == "first":
            embeddings = last_hidden_state
            output = embeddings[:,0,:]
            output = self.output(output)
        elif self.pooling_type == "cross-attention":
            assert output_queries
            output = self.output(last_hidden_state, last_hidden_state, output_queries)
        elif self.pooling_type == "mean-token":
            output = self.output(last_hidden_state[:,:self.ntoken,:]).squeeze()
        return output



class PretrainedEncoder(nn.Module):
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


class PretrainedHyenaEncoder(nn.Module):
    def __init__(self, model_path, freeze_layers = 5, pooling_type="None", n_classes=128, **kwargs):
        """

        :param model_path:
        :param freeze_layers:
        :param pooling_type:
        :param n_classes:
        :param kwargs:
        """
        super().__init__()
        self.model_type = "PretrainedEncoder"
        self.model = HyenaDNAPreTrainedModel.from_pretrained(
                                                    './hyena-dna/checkpoints',
                                                    model_path, #'hyenadna-small-32k-seqlen',
                                                    download=True,
                                                    config=None,
                                                    #device="cpu",
                                                    use_head=True if pooling_type == "hyena" else False,
                                                    n_classes=n_classes) #Not used except for classification
        if freeze_layers is not None:
            print(f"Freezing {freeze_layers} layers")
            modules_to_freeze = [self.model.backbone.embeddings,
                                     self.model.backbone.layers[:freeze_layers]]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
        self.pooling_type = pooling_type
        self.n_classes = n_classes
    def unfreeze_layers(self, layers):
        raise NotImplementedError
        #return
    def forward(self, **kwargs):
        output = self.model(**kwargs)
        if self.pooling_type == "mean":
            output = output.mean(dim=1).squeeze()
        elif self.pooling_type == "first":
            output = output[:,0:self.n_classes,:]
        elif self.pooling_type == "last":
            output = output[:,-self.n_classes:,:]
        return output


class PerturbModel(nn.Module):
    def __init__(self, chem_config, cell_config, decoder_config, loss_config, gene_config=None):
        super().__init__()
        self.model_type = "PerturbationModel"
        self.chem_backbone = PretrainedEncoder(**chem_config)
        self.cell_backbone = PretrainedEncoder(**cell_config)
        if gene_config:
            if gene_config['type'] == "embedding":
                self.gene_backbone = nn.Embedding(gene_config['ntoken'],gene_config['ninp'],max_norm=1.0)
            self.gene_backbone = PretrainedEncoder(**gene_config)
        else:
            self.gene_backbone = None
        self.dosage_backbone = nn.Linear(1, chem_config.hidden_size) if chem_config.embed_dosage else None

        if decoder_config.type == "TransformerDecoder":
            self.decoder_type = "TransformerDecoder"
            self.decoder = TransformerDecoderModel(**decoder_config)
            chem_size = self.chem_backbone.model.encoder.layer[-1].output.dense.out_features
            cell_size = self.cell_backbone.model.encoder.layer[-1].output.dense.out_features
            self.cell_projector = nn.Linear(cell_size, decoder_config.ninp)
            self.chem_projector = nn.Linear(chem_size, decoder_config.ninp)
        elif decoder_config.type == "MLP":
            self.decoder_type = "MLP"
            self.decoder = MLP(**decoder_config)
        #self.return_token = nn.Parameter(torch.randn(1, decoder_config.ninp))

        if loss_config.type == "perturbation":
            self.loss_type = "perturbation"
            print("PerturbationLoss")
            self.loss_fct = PerturbationLoss(**loss_config)
        elif loss_config.type == "mse":
            self.loss_type = "mse"
            self.loss_fct = nn.MSELoss()
        

    def forward(
        self,
        labels,
        chem_batch,
        cell_batch,
        gene_batch=None,
        dosage = None,
        variance=None,
        **kwargs,
    ):

        chem_embedding = self.chem_backbone(**chem_batch)
        cell_embedding = self.cell_backbone(**cell_batch)
        gene_embedding = self.gene_backbone(**gene_batch) if gene_batch else None #Gene batch is just nn.embedding indices right now

        if self.dosage_backbone:
            dosage_embedding = self.dosage_backbone(dosage.unsqueeze(1))
            if chem_embedding.dim() == 3:
                dosage_embedding = dosage_embedding.unsqueeze(1) # for broadcasting
            chem_embedding = chem_embedding * dosage_embedding
            #print(chem_embedding.shape)
        if self.decoder_type == "TransformerDecoder":
            #cell_embedding.last_hidden_state = repeat(cell_embedding.last_hidden_state, 'b t e -> b t (c e)', c=3)
            cell_embedding = self.cell_projector(cell_embedding)
            chem_embedding = self.chem_projector(chem_embedding)
            logits = self.decoder(cell_embedding,
                         chem_embedding,
                         cell_batch['attention_mask']==0,
                         chem_batch['attention_mask']==-1,
                         output_queries=gene_embedding) #Only for pooling_type: cross-attention
        elif self.decoder_type == "MLP":
            embeddings, ps = pack(( chem_embedding, cell_embedding), 'b *')
            logits = self.decoder(embeddings)
        if self.loss_type == "perturbation":
            loss = self.loss_fct(logits, labels, variance)
        elif self.loss_type == "mse":
            loss = self.loss_fct(logits, labels)

        return loss, logits

class SmallMoleculeModel(nn.Module):
    def __init__(self, chem_config, decoder_config, loss_config):
        super().__init__()
        self.model_type = "SmallMoleculeModel"
        self.chem_backbone = PretrainedEncoder(**chem_config)

        if decoder_config.type == "MLP":
            self.decoder_type = "MLP"
            self.decoder = MLP(**decoder_config)
        elif decoder_config.type == "Linear":
            self.decoder_type = "Linear"
            self.decoder = nn.Linear(in_features = decoder_config['ninp'],
                                       out_features = decoder_config['ntoken'])

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
            chem_batch,
            **kwargs,
    ):

        chem_embedding = self.chem_backbone(**chem_batch)
        if self.loss_fct:
            logits = self.decoder(chem_embedding).squeeze()
        
            loss = self.loss_fct(logits, labels.float())

            return loss, F.sigmoid(logits)
        else:
            return chem_embedding

