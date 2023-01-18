# Introduction

We introduction how to run ClusterNS and reproduce the main results in this document.

# Preparation

Before running the code, you may need to implement some preparation, including:
* Download unsupervised training dataset in *data/download_wiki.sh*
* Download testing dataset in *SentEval/data/downstream/download_dataset.sh*
* Prepare the python environment according to *requirements.txt*

# Main Results

We provide two bash scripts for running the code with *Non-prompt* and *Prompt-based* ClusterNS, respectively and release some checkpoints of our models. We release two of them in google drive first due to the limitation of storage space, and all models will be released in Huggingface after ACL2023.
|              Model              | Avg. STS |
|:-------------------------------|:--------:|
|  ClusterNS-non-prmt-bert-base-uncased |   77.55 |
| ClusterNS-non-prmt-bert-large-uncased |   79.19  |
|    [ClusterNS-non-prmt-roberta-base](https://drive.google.com/file/d/17qwll3YI55-5IeZJQdm4yfFSYeCamBco/view?usp=share_link)    |   77.98  |
|    ClusterNS-non-prmt-roberta-large  |   79.43  |
|   ClusterNS-prmt-bert-base-uncased|   78.72  |
|  ClusterNS-prmt-bert-large-uncased  |   80.30  |
|     [ClusterNS-prmt-roberta-base](https://drive.google.com/file/d/1VmzM2LdWzx6SAU4u69tRAsBXEGLbGK5g/view?usp=share_link)     |   79.74  |

# Hyperparameters

We provide some importance hyperparameters in our models for reproducing. Note that we use distributed data parallel for some of model trainings, thus 128*4 in batch_size means **per_device_train_batch_size**=128 with 4 GPUs.
 Hyperparameter | BERT-base | BERT-large | RoBERTa-base | RoBERTa-large | prompt-BERT-base | prompt-BERT-large | prompt-RoBERTa-base  
----------------|-----------|------------|--------------|---------------|------------------|-------------------|----------------------
 batch_size     | 256       | 128*4      | 256*2        | 128*4         | 512              | 64*4              | 256*2                
 model_lr       | 5e-5  | 5e-5   | 5e-5     | 5e-5      | 5e-5         | 3e-5          | 5e-5             
 kmeans         | 128       | 128        | 128          | 128           | 96               | 96                | 128                  
 kmeans_lr      | 1e-3  | 1e-3   | 1e-3     | 5e-4      | 5e-4         | 1e-4          | 1e-3             
 kmean_cosine   | 0.4       | 0.5        | 0.4          | 0.4           | 0.4              | 0.2               | 0.4                  
 enable_hardneg | TRUE      | TRUE       | TRUE         | TRUE          | TRUE             | TRUE              | FALSE                
 bml_weight     | 1e-3  | 5e-5   | 1e-4     | 0             | 1e-2         | 5e-4          | 1e-2             
 bml_alpha      | 0         | 0.25       | 0.2          |               | 0                | 0.25              | 0.2                  
 bml_beta       | 0.3       | 0.3        | 0.5          |               | 0.2              | 0.3               | 0.6                  

