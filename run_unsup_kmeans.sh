#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
NUM_GPU=2
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=2,3
# init-${k}_${cos}_${lr}
#-m torch.distributed.launch --nproc_per_node $NUM_GPU
for lr in 2e-5 5e-5
do
    for cos in 0.2 0.3 0.4 0.5
    do
        for k in 64 128 256
        do
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -m torch.distributed.launch --nproc_per_node $NUM_GPU train.py \
        --model_name_or_path ../plm/roberta-base \
        --train_file data/wiki1m_for_simcse.txt \
        --output_dir result/unsup-roberta-base-momentum-lr${lr}_${cos}_k${k} \
        --cache_dir "dataset_cache" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 256 \
        --gradient_accumulation_steps 1 \
        --save_steps 125 \
        --save_total_limit 2 \
        --learning_rate ${lr} \
        --max_seq_length 32 \
        --evaluation_strategy steps \
        --metric_for_best_model stsb_spearman \
        --load_best_model_at_end \
        --eval_steps 125 \
        --pooler_type cls \
        --mlp_only_train \
        --overwrite_output_dir \
        --do_train \
        --kmeans \
        --k ${k} \
        --kmeans_lr 1e-3 \
        --kmean_cosine ${cos} \
        --kmean_debug \
        --kmeans_optim "momentum" \
        --fp16 \
        "$@"
    #nohup python evaluation.py \
    nohup python evaluation.py \
    --model_name_or_path result/unsup-roberta-base-momentum-lr${lr}_${cos}_k${k} \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test > log_drop_prob/kmeans_momentum_lr${lr}_${cos}_k${k}.log 2>&1
        done
    done
done
#temp = {bert:1, roberta:0.01}
    # --do_mask \
    # --dropping_method high \
#memory with bsz: {ad-drop+mlm:32, ad-drop:64, mlm:64}

what whtat