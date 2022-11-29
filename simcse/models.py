from copy import deepcopy
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tensorboardX import SummaryWriter
from transformers import RobertaTokenizer, AdamW
from transformers.activations import gelu
from transformers.file_utils import (add_code_sample_docstrings,
                                     add_start_docstrings,
                                     add_start_docstrings_to_model_forward,
                                     replace_return_docstrings)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput)
from transformers.models.bert.modeling_bert import (BertLMPredictionHead,
                                                    BertModel,
                                                    BertPreTrainedModel)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead, RobertaModel, RobertaPreTrainedModel)

from .drop_utils import (generate_mask_high, generate_mask_low,
                         generate_mask_random, compute_attention_attribution)


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class ProjectionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        ]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        B, N = x.shape[0], x.shape[1]
        y = x.view((B*N, x.size(-1)))
        y = self.net(y)
        y = y.view((B, N, x.size(-1)))
        return y

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None, #数据组织形式:batch[0,0,1,1....127,127]
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    do_mask=False,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)
    anc_input_ids = input_ids[:,0,:].clone().detach() #experimental stop-gradient
    anc_attention_mask = attention_mask[:,0,:]

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0)) #[bsz/2, 1, nh] * [1, bsz/2, nh] -> [bsz/2, bsz/2]

    #consistency loss
    # cst_loss_fct = nn.SmoothL1Loss(beta=cls.model_args.huber_delta)
    # cst_loss_fct = nn.KLDivLoss()
    # normalized_cos = cos_sim * cls.model_args.temp
    # p1 = torch.softmax(normalized_cos, dim=-1)
    # p2 = torch.softmax(normalized_cos.T, dim=-1)
    # target_p = (p1+p2)/2
    # cst_loss1 = cst_loss_fct(normalized_cos, normalized_cos.T) #* cls.model_args.temp
    # cst_loss2 = cst_loss_fct(normalized_cos.T, normalized_cos) #* cls.model_args.temp
    # cst_loss = (cst_loss1 + cst_loss2) / 2
    cst_loss = None
    adapt_weight = None
    fn_loss = None

    #kmeans clustering
    if cls.model_args.kmeans > 0:
        fn_loss = None
        normalized_cos = cos_sim * cls.model_args.temp
        avg_cos = normalized_cos.mean().item() 
        # z12 = torch.cat([z1, z2], dim=0)
        if not cls.cluster.initialized:
            if avg_cos <= cls.model_args.kmean_cosine:
                # with torch.no_grad():
                #     all_cos_sim = cls.sim(z12.unsqueeze(1), z12.unsqueeze(0))
                cls.cluster.optimized_centroid_init(z1, cos_sim*cls.model_args.temp)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("kmeans start!!")
        elif cls.cluster.initialized:
            # if cls.cluster.global_step % 100 == 0:
            #     if not dist.is_initialized() or dist.get_rank() == 0:
            #         print(cls.cluster.centroid.data[0][:4].tolist())
            cls.cluster(z1, anc_input_ids, normalized_cos)
            num_sent = 3 #to be fix
            z3 = cls.cluster.provide_hard_negative(z1)
            cos_sim_mask = cls.cluster.mask_false_negative(z1, normalized_cos)
            fn_loss = cls.cluster.false_negative_loss(z1, cos_sim_mask, normalized_cos, z3)
            # adapt_weight = cls.cluster.weighted_negative(z1,z2,z3,cos_sim_mask,True)
            # cst_loss = cls.cluster.cluster_consistency_loss(z1, z2)
            # cos_sim_mask = cos_sim_mask==0
            # cos_sim = cos_sim + cos_sim_mask * -10000
            # cos_sim = cos_sim * cos_sim_mask.float()
        cls.cluster.global_step += 1      

    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)  #to be fix      

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights      

    if adapt_weight is not None:
        cos_sim = cos_sim + adapt_weight

    loss = loss_fct(cos_sim, labels)
    if fn_loss is not None:
        loss = loss + cls.model_args.bml_weight * fn_loss
    if cst_loss is not None: #consistency loss
        loss = loss + cls.model_args.cst_weight * cst_loss

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        if self.model_args.dropout_prob is not None:
            config.attention_probs_dropout_prob = self.model_args.dropout_prob
            config.hidden_dropout_prob = self.model_args.dropout_prob           
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        if self.model_args.kmeans > 0:
            self.cluster = kmeans_cluster(config, self.model_args)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        if self.model_args.do_mask:
            config.output_attentions = True
        if self.model_args.dropout_prob is not None:
            config.attention_probs_dropout_prob = self.model_args.dropout_prob
            config.hidden_dropout_prob = self.model_args.dropout_prob           
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)
        if self.model_args.my_debug:
            self.cmpr = []
            tfb_name = "unsup-simcse-cosine" 
            self.writer = SummaryWriter(os.path.join('runs', tfb_name))
        if self.model_args.kmeans > 0:
            self.cluster = kmeans_cluster(config, self.model_args)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

class hook_for_ADdrop:
    def __init__(self, num_layers) -> None:
        self.attn_mask = []
        self.num_layers = num_layers
        self.iterator = 0

    def __call__(self, model, inputs):
        if self.attn_mask:
            #assert len(self.attn_mask) == self.num_layers, "length not equal between model.attn_mask %d and mask_layers %d" % (len(self.attn_mask), self.num_layers) 
            assert len(self.attn_mask), "empty attention mask with self.attn_mask"
            extra_attn_mask = self.attn_mask.pop(0)
            #还原mask:新旧数值对比{0:1,-10000:0}
            original_attn_mask = inputs[1]
            ori_raw_attn_mask = (original_attn_mask == 0)
            #两种mask相加
            extend_attn_mask = ori_raw_attn_mask.expand(extra_attn_mask.shape)
            new_raw_attn_mask = extra_attn_mask * extend_attn_mask
            #mask数值转换
            new_attn_mask = (1.0 - new_raw_attn_mask) * -10000.0
            
            oringinal_outputs = list(inputs)
            oringinal_outputs[1] = new_attn_mask
            new_inputs = tuple(oringinal_outputs)
            self.iterator += 1

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                do_mask=self.model_args.do_mask
            )

class kmeans_cluster(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()
        self.model_args = model_args
        self.k = model_args.kmeans
        self.sim = Similarity(temp=1)
        self.initialized = False
        self.global_step = 0
        self.tokenizer = RobertaTokenizer.from_pretrained(config._name_or_path)
        self.optimization = model_args.kmeans_optim
        self.lr = model_args.kmeans_lr
        self.beta = model_args.bml_beta
        self.alpha = model_args.bml_alpha
        self.huber_delta = model_args.huber_delta
        self.cst_temp = model_args.cst_temp
        self.proto_smooth = model_args.proto_smooth
        if model_args.kmean_debug:
            if not dist.is_initialized() or dist.get_rank() == 0:
                # self.writer = SummaryWriter("runs/kmeans-momentum-lr%.3f-%.1f-%d" % (model_args.logging_lr, model_args.kmean_cosine, model_args.k))
                # self.writer = SummaryWriter("runs/kmeans-bml-w%.1e-b%.2f" % (model_args.bml_weight, model_args.bml_beta))
                self.writer = SummaryWriter("runs/kmeans-7798")

    def centroid_init(self, centroid:torch.Tensor):
        data=centroid.clone().detach()
        idx = list(range(data.shape[0]))
        random.shuffle(idx)
        self.centroid = nn.Parameter(data=data[idx[:self.k]])
        self.optimizer = AdamW(self.parameters(), lr=self.lr)
        self.initialized = True

    def provide_hard_negative(self, datapoints:torch.Tensor, batch_cos_sim:torch.Tensor=None):
        D = self.centroid.data.shape[-1]
        if batch_cos_sim is None:
            batch_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        values, indices = torch.topk(batch_cos_sim, k=2, dim=-1)
        hard_neg_index = indices[:, 1].unsqueeze(-1).expand(-1, D)
        hard_negative = torch.gather(self.centroid, dim=0, index=hard_neg_index)
        return hard_negative.detach()

    def mask_false_negative(self,
                            datapoints:torch.Tensor, 
                            batch_cos_sim:torch.Tensor=None, #cos(xi, xj) [bsz, bsz]
    ):   
        if batch_cos_sim is None:
            batch_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        intra_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        dp_centroid_cos, dp_index = torch.max(intra_cos_sim, dim=-1, keepdim=True) #(bsz, 1)
        dp_cluster, _ = self.intra_class_adjacency_(dp_index) #(bsz, bsz)
        dp_centroid_cos = dp_centroid_cos.expand_as(dp_cluster) #(bsz, bsz)

        # false_negative_mask: {1:masked, 0:unmasked}
        # false_negative_mask = dp_cluster * (batch_cos_sim>dp_centroid_cos)
        false_negative_mask = dp_cluster 
        return false_negative_mask

    def intra_class_adjacency_(self, dp_index:torch.Tensor):
        r'''
        dp_index: indicating the indices of cluster to which datapoints belong

        return:
        dp_cluster: [bsz, bsz], indicating that which datapoints belong to same cluster
        index_dp: [k, bsz], indicating that which datapoints belong to the clusters
        '''

        B, device = dp_index.shape[0], dp_index.device
        onehot_index = F.one_hot(dp_index.squeeze(-1), self.k) #(bsz, k)
        index_dp = onehot_index.T #(k, bsz)
        
        #adjacency matrix that dp_cluster[i][j]==1 if xi and xj belong to identical cluster
        dp_cluster = torch.matmul(onehot_index.float(), index_dp.float()) 
        #set dp_cluster[i][i] = 0
        dp_cluster.fill_diagonal_(0) 
        return dp_cluster, index_dp

    def false_negative_loss(self,
                            datapoints:torch.Tensor,
                            batch_false_negative_mask:torch.Tensor,
                            batch_cos_sim:torch.Tensor=None,
                            batch_hard_negative:torch.Tensor=None,
                            reduction:str="mean", 
                            alpha:torch.Tensor=None, 
                            beta=None,
    ):
        """
        use BML loss for false negative examples. only implemented example-level now.

        batch_cos_sim: [bs, bs] 
        batch_false_negative_mask: [bs, bs], {1:masked, 0:unmasked}
        batch_hard_negative: [bs, bs]
        """
        if batch_cos_sim is None:
            batch_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        batch_positive_sim = batch_cos_sim.diag().unsqueeze(-1).expand_as(batch_cos_sim)
        if batch_hard_negative is None:
            batch_hard_negative = self.provide_hard_negative(datapoints, batch_cos_sim)
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            # our goal:
            # cos(x, hard_neg) < cos(x, false_neg) < cos(x, pos)
            # cos(x, hard_neg)-cos(x, pos) < cos(x, false_neg)-cos(x, pos) < 0
            # -beta < cos(x, false_neg)-cos(x, pos) < -alpha
            # dp_hardneg_sim = self.sim(datapoints.unsqueeze(1), batch_hard_negative.unsqueeze(0))
            # batch_hardneg_sim = dp_hardneg_sim.diag().unsqueeze(-1).expand_as(batch_cos_sim)
            # beta = batch_hardneg_sim - batch_positive_sim 
            beta = self.beta
        batch_delta = (batch_cos_sim - batch_positive_sim) * batch_false_negative_mask
        loss = self.BML_loss(batch_delta, alpha, beta)
        if reduction == "mean":
            return loss.mean()
        else:
            raise NotImplementedError

    @staticmethod
    def BML_loss(x, alpha, beta):
        """use for example-level BML loss"""
        return F.relu(x + alpha) + F.relu(-x - beta)

    def weighted_negative(self,
        z1:torch.Tensor,
        z2:torch.Tensor,
        z3:torch.Tensor=None,
        cos_sim_mask:torch.Tensor=None,
        inverse=False,
    ):
        """
        batch_center_points: [bs, hidden_size], 每个锚点所属类的类中心向量
        batch_hard_neg_center_points: [bs, hidden_size], 每个锚点难负例所属类的类中心向量
        inverse: 如果为 True，类中心余弦相似度大的，权重小；
        return: weight for loss, should add this to cos_sim
        """
        B, device = z1.shape[0], z1.device
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        if cos_sim_mask is not None:
            cos_sim += cos_sim_mask * -10000
        if z3 is not None:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
        # weights = F.cosine_similarity(batch_center_points, batch_center_points)  # (bs, bs)
        # if batch_hard_neg_center_points:
        #     neg_weight = F.cosine_similarity(batch_center_points, batch_hard_neg_center_points)  # (bs, bs)
        #     weights = torch.cat([weights, neg_weight], dim=-1)  # (bs, 2bs)
        if inverse is True:
            cos_sim = -cos_sim
        cos_sim.fill_diagonal_(-10000)
        weights = F.softmax(cos_sim, dim=-1) / 0.05
        return weights.detach()
    
    def cluster_consistency_loss(self, z1:torch.Tensor, z2:torch.Tensor):
        centroid = self.centroid.data.clone().detach()
        z1_cos_sim = self.sim(z1.unsqueeze(1), centroid.squeeze(0)) / self.cst_temp
        z2_cos_sim = self.sim(z2.unsqueeze(1), centroid.squeeze(0)) / self.cst_temp

        #consistency loss
        cst_loss_fct = nn.KLDivLoss()
        p1 = torch.softmax(z1_cos_sim, dim=-1)
        p2 = torch.softmax(z2_cos_sim, dim=-1)        
        cst_loss1 = cst_loss_fct(p2, p1) #
        # cst_loss2 = cst_loss_fct(p2, p1) #* cls.model_args.temp
        cst_loss = cst_loss1
        return cst_loss
    
    def new_cluster_consistency_loss(self, z1:torch.Tensor, z2:torch.Tensor):
        centroid = self.centroid.data.clone().detach()
        z1_cos_sim = self.sim(z1.unsqueeze(1), centroid.squeeze(0)) / self.cst_temp
        z2_cos_sim = self.sim(z2.unsqueeze(1), centroid.squeeze(0)) 

        labels = z1_cos_sim.argmax(dim=-1)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(z1_cos_sim, labels)
        return loss

    def ProtoNCE_loss(self, datapoints:torch.Tensor):
        B, D = datapoints.shape
        intra_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        dp_centroid_cos, dp_index = torch.max(intra_cos_sim, dim=-1, keepdim=True) #(bsz, 1)
        _, index_dp = self.intra_class_adjacency_(dp_index) #(k, bsz)
        extend_dp = datapoints.unsqueeze(0).expand(self.k, B, -1) #(k, bsz, hid)
        extend_centroid = self.centroid.data.unsqueeze(1).expand(self.k, B, -1) #(k, bsz, hid)

        # average distance between centroid and datapoints belong to the cluster, empty cluster will set to 0
        L2_distance = nn.PairwiseDistance()
        dist_dp_centroid = L2_distance(extend_dp.reshape(-1, D), extend_centroid.reshape(-1, D)) #(k*B)
        dist_dp_centroid = dist_dp_centroid.reshape(self.k, B) #(k, B)
        l2_dist = torch.sum(index_dp * dist_dp_centroid, dim=-1)
        member_count = index_dp.sum(dim=-1)
        smooth_member_count = torch.where(member_count==0, 1, member_count)
        avg_l2_dist = l2_dist / smooth_member_count

        # calculate phi for each cluster
        centroid_phi = avg_l2_dist / torch.log(smooth_member_count+self.proto_smooth) #use smooth_member_count to avoid empty cluster become inf
        centroid_phi = centroid_phi.unsqueeze(0).expand(B, self.k)
        zero_mask = torch.where(member_count==0, -10000, 0)
        zero_mask = zero_mask.unsqueeze(0).expand(B, self.k).float() #for masking empty cluster
        smooth_cos_sim = torch.addcdiv(zero_mask, intra_cos_sim, centroid_phi+zero_mask)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(smooth_cos_sim, dp_index.squeeze(-1))
        return loss

    
    def forward(self, 
                datapoints:torch.Tensor, 
                input_ids:torch.Tensor, 
                batch_cos_sim:torch.Tensor, 
                step:int=None, 
                writer:SummaryWriter=None
    ):
        B = datapoints.shape[0]
        device = datapoints.device
        datapoints = datapoints.clone().detach()
        intra_cos_sim = self.sim(datapoints.unsqueeze(1), self.centroid.unsqueeze(0))
        dp_index = torch.argmax(intra_cos_sim, dim=-1, keepdim=True) #(bsz, 1)
        dp_cluster, index_dp = self.intra_class_adjacency_(dp_index) #(bsz, bsz)

        dp_centroid = torch.gather(self.centroid, dim=0, index=dp_index.expand_as(datapoints)) #set the centroid corresponding to the datapoint
        if self.model_args.kmean_debug:
            if not dist.is_initialized() or dist.get_rank() == 0:
                dp_hardneg = self.provide_hard_negative(datapoints, intra_cos_sim)
                self.debug_stat(datapoints, dp_centroid, dp_hardneg, batch_cos_sim, dp_index.squeeze(-1), dp_cluster)

        # if self.model_args.my_debug:
        # sentence = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # all_datapoint = torch.cat((self.centroid.data, datapoints)).detach().cpu().numpy()
        # metalist = ["centroid-%d"%i for i in range(self.k)]
        # for i, sent in enumerate(sentence):
        #     metalist.append("class-%d-instance-%s"%(dp_index[i][0], sent))
        # self.writer.add_embedding(all_datapoint, metadata=metalist, global_step=self.global_step)

        # loss_fct = nn.MSELoss()
        # kmeans_loss = loss_fct(dp_centroid, datapoints) 
        # kmeans_loss.backward()  
        # self.optimizer.step()
        # self.optimizer.zero_grad()   
        # return kmeans_loss
        self.update(datapoints, dp_centroid, index_dp)

    def debug_stat(
        self, 
        datapoints:torch.Tensor, 
        dp_centroid:torch.Tensor, 
        dp_hardneg:torch.Tensor,
        batch_cos_sim:torch.Tensor,
        dp_index:torch.Tensor,
        dp_cluster:torch.Tensor,
    ):
        with torch.no_grad():
            device = datapoints.device
            diag_mask = torch.diag(torch.ones(self.k)).to(device)
            diag_mask = 1 - diag_mask

            #Cosine Distance
            positive_avg_cos = batch_cos_sim.diag().mean()
            batch_avg_cos = batch_cos_sim.mean()
            centroid_cosine = self.sim(self.centroid.unsqueeze(1), self.centroid.unsqueeze(0))
            centroid_cosine *= diag_mask
            centroid_avg_cos = centroid_cosine.sum() / (self.k*(self.k-1))
            cent_dp_cosine = self.sim(datapoints.unsqueeze(1), dp_centroid.unsqueeze(0))
            cent_dp_max_cos = cent_dp_cosine.diag().max()
            cent_dp_min_cos = cent_dp_cosine.diag().min()
            cent_dp_avg_cos = cent_dp_cosine.diag().mean()
            hard_neg_cosine = self.sim(datapoints.unsqueeze(1), dp_hardneg.unsqueeze(0))
            hard_neg_max_cos = hard_neg_cosine.diag().max()
            hard_neg_min_cos = hard_neg_cosine.diag().min()
            hard_neg_avg_cos = hard_neg_cosine.diag().mean()

            member_cos_sim = batch_cos_sim * dp_cluster
            member_mask = torch.logical_not(dp_cluster).float()
            avg_member_cos = member_cos_sim.sum() / dp_cluster.count_nonzero()
            max_member_cos = member_cos_sim.max()
            min_member_cos = (member_cos_sim + member_mask).min()

            cosine_dict = {
                "inter-centroid": centroid_avg_cos, "intra-cos": cent_dp_avg_cos, "hn-cos": hard_neg_avg_cos,
                "positive-cos": positive_avg_cos, "batch-cos": batch_avg_cos, "member-cos": avg_member_cos
            }
            self.writer.add_scalars("cosine", cosine_dict, self.global_step)
            detail_cosine = {
                "max-intra-cos": cent_dp_max_cos, "min-intra-cos": cent_dp_min_cos,
                "max-hardneg-cos": hard_neg_max_cos, "min-hardneg-cos": hard_neg_min_cos,
                "max-member-cos": max_member_cos, "min-member-cos": min_member_cos
            }
            self.writer.add_scalars("detail_cosine", detail_cosine, self.global_step)

            #Euclidean Distance
            euc_distance = nn.PairwiseDistance()
            centroid_L2dist = torch.cdist(self.centroid, self.centroid)
            centroid_L2dist *= diag_mask
            centroid_avg_L2 = centroid_L2dist.sum() / (self.k*(self.k-1))
            cent_dp_L2dist = euc_distance(datapoints, dp_centroid)
            cent_dp_max_L2 = cent_dp_L2dist.max()
            L2dist_dict = {"inter-centroid": centroid_avg_L2, "max-intra-euc": cent_dp_max_L2}
            self.writer.add_scalars("euc_distance", L2dist_dict, self.global_step)

            #how many member in each cluster
            cluster_count = torch.zeros((self.k))

            for n in dp_index:
                try:
                    cluster_count[n] += 1
                except:
                    if dist.is_initialized() and dist.get_rank() == 0:
                        import pdb; pdb.set_trace()                    
            avg_cluster_member = cluster_count.sum() / cluster_count.count_nonzero()
            max_cluster_member = cluster_count.max()
            cluster_dict = {"avg_member": avg_cluster_member, "max_member": max_cluster_member}
            self.writer.add_scalars("cluster_member", cluster_dict, self.global_step)

    def optimized_centroid_init(self, centroid:torch.Tensor, batch_cos_sim:torch.Tensor):
        data=centroid.clone().detach()
        L = data.shape[0]
        assert self.k <= L
        self.centroid = nn.Parameter(data=torch.zeros_like(data[:self.k]))
        idx = list(range(data.shape[0]))  
        first_idx = random.randint(0, L-1)    
        self.centroid.data[0] = data[first_idx]
        last_idx = first_idx
        
        for i in range(1, self.k):
            #set the last centroid in cos_sim to maxmimal, it will be ignored in the later centroid selection process.
            batch_cos_sim[:, last_idx] = 100
            next_idx = torch.argmin(batch_cos_sim[last_idx])
            self.centroid.data[i] = data[next_idx]
            last_idx = next_idx
        
        self.optimizer = AdamW(self.parameters(), lr=self.lr)
        self.initialized = True

    def update(
        self, 
        datapoints:torch.Tensor, 
        dp_centroid:torch.Tensor=None,
        index_dp:torch.Tensor=None, #[k, bsz]
    ):
        #update self.centroid in various ways
        if self.optimization == "adamw":
            loss_fct = nn.MSELoss()
            kmeans_loss = loss_fct(dp_centroid, datapoints) 
            kmeans_loss.backward()  
            self.optimizer.step()
            self.optimizer.zero_grad()       
        elif self.optimization in ["kmeans", "momentum"]:
            data = torch.matmul(index_dp.float(), datapoints)
            updated_centroid = index_dp.sum(dim=-1).bool().unsqueeze(1).expand_as(self.centroid.data)
            data += self.centroid.data * updated_centroid.logical_not()
            if self.optimization == "kmeans":
                self.centroid.data = data
            elif self.optimization == "momentum":
                old_data = self.centroid.data
                self.centroid.data = self.lr * data + (1-self.lr) * old_data
