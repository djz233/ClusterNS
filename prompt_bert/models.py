import random
from re import template
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from tensorboardX import SummaryWriter
from transformers import RobertaTokenizer, AdamW
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config, scale=1):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*scale, config.hidden_size*scale)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

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



def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    if cls.model_args.mask_embedding_sentence_org_mlp:
        from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
        cls.mlp = BertPredictionHeadTransform(config)
    else:
        cls.mlp = MLPLayer(config, scale=cls.model_args.mask_embedding_sentence_num_masks)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
               encoder,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               output_attentions=None,
               output_hidden_states=None,
               labels=None,
               return_dict=None,
):
    def get_delta(template_token, length=50):
        with torch.set_grad_enabled(not cls.model_args.mask_embedding_sentence_delta_freeze):
            device = input_ids.device
            d_input_ids = torch.Tensor(template_token).repeat(length, 1).to(device).long()
            if cls.model_args.mask_embedding_sentence_autoprompt:
                d_inputs_embeds = encoder.embeddings.word_embeddings(d_input_ids)
                p = torch.arange(d_input_ids.shape[1]).to(d_input_ids.device).view(1, -1)
                b = torch.arange(d_input_ids.shape[0]).to(d_input_ids.device)
                for i, k in enumerate(cls.dict_mbv):
                    if cls.fl_mbv[i]:
                        index = ((d_input_ids == k) * p).max(-1)[1]
                    else:
                        index = ((d_input_ids == k) * -p).min(-1)[1]
                    #print(d_inputs_embeds[b,index][0].sum().item(), cls.p_mbv[i].sum().item())
                    #print(d_inputs_embeds[b,index][0].mean().item(), cls.p_mbv[i].mean().item())
                    d_inputs_embeds[b, index] = cls.p_mbv[i]
            else:
                d_inputs_embeds = None
            d_position_ids = torch.arange(d_input_ids.shape[1]).to(device).unsqueeze(0).repeat(length, 1).long()
            if not cls.model_args.mask_embedding_sentence_delta_no_position:
                d_position_ids[:, len(cls.bs)+1:] += torch.arange(length).to(device).unsqueeze(-1)
            m_mask = d_input_ids == cls.mask_token_id
            outputs = encoder(input_ids=d_input_ids if d_inputs_embeds is None else None ,
                              inputs_embeds=d_inputs_embeds,
                              position_ids=d_position_ids,  output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            delta = last_hidden[m_mask]
            template_len = d_input_ids.shape[1]
            if cls.model_args.mask_embedding_sentence_org_mlp:
                delta = cls.mlp(delta)
            return delta, template_len

    if cls.model_args.mask_embedding_sentence_delta:
        delta, template_len = get_delta([cls.mask_embedding_template])
        if len(cls.model_args.mask_embedding_sentence_different_template) > 0:
            delta1, template_len1 = get_delta([cls.mask_embedding_template2])

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    if cls.model_args.mask_embedding_sentence_autoprompt:
        inputs_embeds = encoder.embeddings.word_embeddings(input_ids)
        p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
        b = torch.arange(input_ids.shape[0]).to(input_ids.device)
        for i, k in enumerate(cls.dict_mbv):
            if cls.model_args.mask_embedding_sentence_autoprompt_continue_training_as_positive and i%2 == 0:
                continue
            if cls.fl_mbv[i]:
                index = ((input_ids == k) * p).max(-1)[1]
            else:
                index = ((input_ids == k) * -p).min(-1)[1]
            #print(inputs_embeds[b,index][0].sum().item(), cls.p_mbv[i].sum().item())
            #print(inputs_embeds[b,index][0].mean().item(), cls.p_mbv[i].mean().item())
            inputs_embeds[b, index] = cls.p_mbv[i]

    outputs = encoder(
        None if cls.model_args.mask_embedding_sentence_autoprompt else input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )


    # Pooling
    if cls.model_args.mask_embedding_sentence:
        last_hidden = outputs.last_hidden_state
        pooler_output = last_hidden[input_ids == cls.mask_token_id]

        if cls.model_args.mask_embedding_sentence_delta:
            if cls.model_args.mask_embedding_sentence_org_mlp:
                pooler_output = cls.mlp(pooler_output)

            if len(cls.model_args.mask_embedding_sentence_different_template) > 0:
                pooler_output = pooler_output.view(batch_size, num_sent, -1)
                attention_mask = attention_mask.view(batch_size, num_sent, -1)
                blen = attention_mask.sum(-1) - template_len
                pooler_output[:, 0, :] -= delta[blen[:, 0]]
                blen = attention_mask.sum(-1) - template_len1
                pooler_output[:, 1, :] -= delta1[blen[:, 1]]
                if num_sent == 3:
                    pooler_output[:, 2, :] -= delta1[blen[:, 2]]
            else:
                blen = attention_mask.sum(-1) - template_len
                pooler_output -= delta[blen]

        pooler_output = pooler_output.view(batch_size * num_sent, -1)

    #if cls.model_args.add_pseudo_instances:
        #batch_size *= 2
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.model_args.mask_embedding_sentence_delta and cls.model_args.mask_embedding_sentence_org_mlp:
        # ignore the delta and org
        pass
    else:
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

    if cls.model_args.dot_sim:
        cos_sim = torch.mm(torch.sigmoid(z1), torch.sigmoid(z2.permute(1, 0)))
    else:
        #if cls.model_args.mask_embedding_sentence_whole_vocab_cl:
            #z1, z2 = torch.sigmoid(z1), torch.sigmoid(z2)
        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    fn_loss = None

    #kmeans clustering
    if cls.model_args.kmeans > 0:
        normalized_cos = cos_sim * cls.model_args.temp
        avg_cos = normalized_cos.mean().item() 
        if not cls.cluster.initialized:
            if avg_cos <= cls.model_args.kmean_cosine:
                cls.cluster.optimized_centroid_init(z1, cos_sim*cls.model_args.temp)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("kmeans start!!")
        elif cls.cluster.initialized:
            cls.cluster(z1, normalized_cos)
            if cls.model_args.enable_hardneg:
                num_sent = 3 #to be fix
                z3 = cls.cluster.provide_hard_negative(z1)
            cos_sim_mask = cls.cluster.mask_false_negative(z1, normalized_cos)
            fn_loss = cls.cluster.false_negative_loss(z1, cos_sim_mask, normalized_cos, None)
        cls.cluster.global_step += 1    

    # Hard negative
    if cls.model_args.norm_instead_temp:
        cos_sim *= cls.sim.temp
        cmin, cmax = cos_sim.min(), cos_sim.max()
        cos_sim = (cos_sim - cmin)/(cmax - cmin)/cls.sim.temp

    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    loss_fct = nn.CrossEntropyLoss()
    labels = torch.arange(cos_sim.size(0)).long().to(input_ids.device)

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(input_ids.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)
    if fn_loss is not None:
        loss = loss + cls.model_args.bml_weight * fn_loss

    # Calculate loss for MLM
    # if not cls.model_args.add_pseudo_instances and mlm_outputs is not None and mlm_labels is not None:
    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states if not cls.model_args.only_embedding_training else None,
        attentions=outputs.attentions if not cls.model_args.only_embedding_training else None,
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

    if cls.model_args.mask_embedding_sentence_delta and not cls.model_args.mask_embedding_sentence_delta_no_delta_eval :
        device = input_ids.device
        d_input_ids = torch.Tensor([cls.mask_embedding_template]).repeat(128, 1).to(device).long()
        d_position_ids = torch.arange(d_input_ids.shape[1]).to(device).unsqueeze(0).repeat(128, 1).long()
        if not cls.model_args.mask_embedding_sentence_delta_no_position:
            d_position_ids[:, len(cls.bs)+1:] += torch.arange(128).to(device).unsqueeze(-1)
        m_mask = d_input_ids == cls.mask_token_id

        with torch.no_grad():
            outputs = encoder(input_ids=d_input_ids, position_ids=d_position_ids,  output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            delta = last_hidden[m_mask]
        delta.requires_grad = False
        template_len = d_input_ids.shape[1]

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if cls.model_args.mask_embedding_sentence and hasattr(cls, 'bs'):
        new_input_ids = []
        bs = torch.LongTensor(cls.bs).to(input_ids.device)
        es = torch.LongTensor(cls.es).to(input_ids.device)

        for i in input_ids:
            ss = i.shape[0]
            d = i.device
            ii = i[i != cls.pad_token_id]
            ni = [ii[:1], bs]
            if ii.shape[0] > 2:
                ni += [ii[1:-1]]
            ni += [es, ii[-1:]]
            if ii.shape[0] < i.shape[0]:
                ni += [i[i == cls.pad_token_id]]
            ni = torch.cat(ni)
            try:
                assert ss + bs.shape[0] + es.shape[0] == ni.shape[0]
            except:
                print(ss + bs.shape[0] + es.shape[0])
                print(ni.shape[0])
                print(i.tolist())
                print(ni.tolist())
                assert 0

            new_input_ids.append(ni)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = (input_ids != cls.pad_token_id).long()
        token_type_ids = None

    if cls.model_args.mask_embedding_sentence_autoprompt:
        inputs_embeds = encoder.embeddings.word_embeddings(input_ids)
        with torch.no_grad():
            p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
            b = torch.arange(input_ids.shape[0]).to(input_ids.device)
            for i, k in enumerate(cls.dict_mbv):
                if cls.fl_mbv[i]:
                    index = ((input_ids == k) * p).max(-1)[1]
                else:
                    index = ((input_ids == k) * -p).min(-1)[1]
                inputs_embeds[b, index] = cls.p_mbv[i]

    outputs = encoder(
        None if cls.model_args.mask_embedding_sentence_autoprompt else input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )

    if cls.model_args.mask_embedding_sentence and hasattr(cls, 'bs'):
        last_hidden = outputs.last_hidden_state
        pooler_output = last_hidden[input_ids == cls.mask_token_id]
        if cls.model_args.mask_embedding_sentence_delta and not cls.model_args.mask_embedding_sentence_delta_no_delta_eval :
            blen = attention_mask.sum(-1) - template_len
            if cls.model_args.mask_embedding_sentence_org_mlp and not cls.model_args.mlp_only_train:
                pooler_output, delta = cls.mlp(pooler_output), cls.mlp(delta)
            pooler_output -= delta[blen]

        if cls.model_args.mask_embedding_sentence_avg:
            pooler_output = pooler_output.view(input_ids.shape[0], -1)
        else:
            pooler_output = pooler_output.view(input_ids.shape[0], -1, pooler_output.shape[-1]).mean(1)
    if not cls.model_args.mlp_only_train and not cls.model_args.mask_embedding_sentence_org_mlp:
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
        self.bert = BertModel(config)

        if self.model_args.mask_embedding_sentence_autoprompt:
            # register p_mbv in init, avoid not saving weight
            self.p_mbv = torch.nn.Parameter(torch.zeros(10))
            for param in self.bert.parameters():
                param.requires_grad = False
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
            )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        if self.model_args.dropout_prob is not None:
            config.attention_probs_dropout_prob = self.model_args.dropout_prob
            config.hidden_dropout_prob = self.model_args.dropout_prob           
        self.roberta = RobertaModel(config, add_pooling_layer=False)

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
            )

class kmeans_cluster(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()
        self.model_args = model_args
        self.k = model_args.kmeans
        self.sim = Similarity(temp=1)
        self.initialized = False
        self.global_step = 0
        self.optimization = model_args.kmeans_optim
        self.lr = model_args.kmeans_lr
        self.beta = model_args.bml_beta
        self.alpha = model_args.bml_alpha
        if model_args.kmean_debug:
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.writer = SummaryWriter("runs/kmeans-prmpt-7974-hn")


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
        # if batch_hard_negative is None:
        #     batch_hard_negative = self.provide_hard_negative(datapoints, batch_cos_sim)
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
    
    def forward(self, 
                datapoints:torch.Tensor, 
                batch_cos_sim:torch.Tensor, 
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
