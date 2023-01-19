#!/bin/bash

NUM_GPU=2
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1

# template for bert, originate from PromptBERT
TEMPLATE="*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*"
TEMPLATE2="*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*"

# template for roberta, originate from PromptBERT
TEMPLATE="*cls*_This_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"
TEMPLATE2="*cls*_The_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"

python -m torch.distributed.launch --nproc_per_node $NUM_GPU train_prompt.py \
    --model_name_or_path ../plm/bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/ClusterNS-prmt-bert-base \
    --num_train_epochs 1 \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --save_steps 125 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --mlp_only_train \
    --overwrite_output_dir \
    --do_train \
    --kmeans 128 \
    --kmean_cosine 0.4 \
    --enable_hardneg \
    --mask_embedding_sentence \
    --mask_embedding_sentence_delta \
    --mask_embedding_sentence_template $TEMPLATE\
    --mask_embedding_sentence_different_template $TEMPLATE2\
    --fp16 \
    --bml_weight 1e-5 \
    --bml_alpha 0.1 \
    --bml_beta 0.3 \
    --early_stop 3 \
    "$@"

python evaluation_prompt.py \
--model_name_or_path result/ClusterNS-prmt-bert-base \
--pooler avg \
--task_set sts \
--mask_embedding_sentence \
--mask_embedding_sentence_template $TEMPLATE \
--mode test 
