import random
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel,
    EarlyStoppingCallback
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from prompt_bert.models import RobertaForCL, BertForCL
from prompt_bert.trainers import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )

    two_bert: bool = field(
        default=False,
        metadata={
        }
    )

    freeze_layers: str= field(
        default='',
        metadata={
        }
    )
    freeze_lm_head: bool = field(
        default=False,
        metadata={
        }
    )
    freeze_embedding: bool = field(
        default=False,
        metadata={
        }
    )
    label_smoothing: bool = field(
        default=False,
        metadata={
        }
    )
    two_bert_one_freeze: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_infomax: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_template: str = field(
        default="*cls*_The_sentence_:_'_*sent_0*_'_means*mask*.*sep+*",
        metadata={
        }
    )
    mask_embedding_sentence_bs: str = field(
        default='This sentence of "',
        metadata={
        }
    )
    mask_embedding_sentence_es: str = field(
        default='" means [MASK].',
        metadata={
        }
    )
    mask_embedding_sentence_with_mlm: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_different_template: str= field(
        default="*cls*_The_sentence_:_'_*sent_0*_'_means*mask*.*sep+*",
        metadata={
        }
    )
    mask_embedding_sentence_delta_freeze: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_delta_cross_stream: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_delta_no_position: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_delta: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_delta_cotrain: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_delta_no_delta_eval: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_do_oxford: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_add_period: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_num_masks: int = field(
        default=1,
        metadata={
        }
    )
    mask_embedding_sentence_avg: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_add_template_in_batch: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt_random_init: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt_continue_training: str= field(
        default='',
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt_continue_training_as_positive: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_autoprompt_freeze_prompt: bool = field(
        default=False,
        metadata={
        }
    )
    mask_embedding_sentence_org_mlp: bool = field(
        default=False,
        metadata={
        }
    )
    only_embedding_training: bool = field(
        default=False,
        metadata={
        }
    )
    roberta_with_special_token: bool = field(
        default=False,
        metadata={
        }
    )
    roberta_auto_weight_special_token: bool = field(
        default=False,
        metadata={
        }
    )
    roberta_special_token_as_cls: bool = field(
        default=False,
        metadata={
        }
    )
    remove_last_layer: bool = field(
        default=False,
        metadata={
        }
    )

    dot_sim: bool = field(
        default=False,
        metadata={
        }
    )

    norm_instead_temp: bool = field(
        default=False,
        metadata={
        }
    )
    add_pseudo_instances: bool = field(
        default=False,
        metadata={
        }
    )
    add_pseudo_instances_from_other_model: bool = field(
        default=False,
        metadata={
        }
    )

    only_negative_loss: bool = field(
        default=False,
        metadata={
        }
    )
    untie_weights_roberta: bool = field(
        default=False,
        metadata={
        }
    )
    token_classification: bool = field(
        default=False,
        metadata={
        }
    )
    add_rdrop: bool = field(
        default=False,
        metadata={
        }
    )
    dropout_prob: Optional[float] = field(
        default=None,
    )
    kmeans: Optional[int] = field(
        default=128,
        metadata={
            "help": "number of cluster in kmeans, -1 mean do not cluster"
            "recommand search space: [32,64,128,256], 2**x"
        }
    )
    kmean_cosine: Optional[float] = field(
        default=0.4,
        metadata={
            "help": "kmeans cluster will start"
            "when the average cosine similarity decrease to kmean_cosine"
            "recommand search space: 0.2~0.6, interval=0.1"
        }
    )
    kmeans_lr: Optional[float] = field(
        default=1e-3,
        metadata={
            "help": "learning rate for kmeans centroid, for both momentum and Adamw"
        }
    )
    kmean_debug: Optional[bool] = field(
        default=False
    )
    logging_lr: Optional[bool] = field(
        default=None
    )
    kmeans_optim: Optional[str] = field(
        default="momentum",
        metadata={
            "help": "[momentum, kmeans, adamw]"
        }
    )
    bml_weight: Optional[float] = field(
        default=1e-4
    )
    bml_beta: Optional[float] = field(
        default=0.5
    )
    bml_alpha: Optional[float] = field(
        default=0.2
    ) 
    enable_hardneg: Optional[bool] = field(
        default=True
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    # reset follow flag type Optional[bool] -> bool
    # to fix typing error for TrainingArguments Optional[bool] in transformers==4.2.1
    # https://github.com/huggingface/transformers/pull/10672
    ddp_find_unused_parameters: bool = field(
        default=None,
        metadata={
            "help": "When using distributed training, the value of the flag `find_unused_parameters` passed to "
            "`DistributedDataParallel`."
        },
    )
    disable_tqdm: bool = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )
    remove_unused_columns: bool = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    greater_is_better: bool = field(
        default=True, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    early_stop: Optional[int] = field(
        default=20,
    )   

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args                  
            )

        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.mask_embedding_sentence_org_mlp:
                from transformers import BertForMaskedLM, BertConfig
                config = BertConfig.from_pretrained(model_args.model_name_or_path)
                model.mlp = BertForMaskedLM.from_pretrained(model_args.model_name_or_path, config=config).cls.predictions.transform

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    column_names = datasets["train"].column_names
    sent2_cname = None
    if model_args.mask_embedding_sentence_do_oxford:
        sent0_cname = column_names[1]
        sent1_cname = column_names[2]
    elif len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    if model_args.mask_embedding_sentence:
        if model_args.mask_embedding_sentence_template != '':
            template = model_args.mask_embedding_sentence_template
            assert ' ' not in template
            template = template.replace('*mask*', tokenizer.mask_token)\
                               .replace('*sep+*', '')\
                               .replace('*cls*', '').replace('*sent_0*', ' ')
            template = template.split(' ')
            model_args.mask_embedding_sentence_bs = template[0].replace('_', ' ')
            if 'roberta' in model_args.model_name_or_path:
                # remove empty block
                model_args.mask_embedding_sentence_bs = model_args.mask_embedding_sentence_bs.strip()
            model_args.mask_embedding_sentence_es = template[1].replace('_', ' ')
        if model_args.mask_embedding_sentence_different_template != '':
            template = model_args.mask_embedding_sentence_different_template
            assert ' ' not in template
            template = template.replace('*mask*', tokenizer.mask_token)\
                               .replace('*sep+*', '')\
                               .replace('*cls*', '').replace('*sent_0*', ' ')
            template = template.split(' ')
            model_args.mask_embedding_sentence_bs2 = template[0].replace('_', ' ')
            if 'roberta' in model_args.model_name_or_path:
                # remove empty block
                model_args.mask_embedding_sentence_bs2 = model_args.mask_embedding_sentence_bs2.strip()
            model_args.mask_embedding_sentence_es2 = template[1].replace('_', ' ')
        #mask_embedding_sentence_bs_tokens = tokenizer.encode(model_args.mask_embedding_sentence_bs, add_special_tokens=False)
        #mask_embedding_sentence_es_tokens = tokenizer.encode(model_args.mask_embedding_sentence_es, add_special_tokens=False)


    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.

        total = len(examples[sent0_cname])

        # Avoid "None" fields
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "

        sentences = examples[sent0_cname] + examples[sent1_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        if model_args.mask_embedding_sentence:
            bs = tokenizer.encode(model_args.mask_embedding_sentence_bs)[:-1]
            es = tokenizer.encode(model_args.mask_embedding_sentence_es)[1:] # remove cls or bos

            if len(model_args.mask_embedding_sentence_different_template) > 0:
                bs2 = tokenizer.encode(model_args.mask_embedding_sentence_bs2)[:-1]
                es2 = tokenizer.encode(model_args.mask_embedding_sentence_es2)[1:] # remove cls or bos
            else:
                bs2, es2 = bs, es

            sent_features = {'input_ids': [], 'attention_mask': []}
            for i, s in enumerate(sentences):
                if i < total:
                    s = tokenizer.encode(s, add_special_tokens=False)[:data_args.max_seq_length]
                    sent_features['input_ids'].append(bs+s+es)
                elif i < 2*total:
                    s = tokenizer.encode(s, add_special_tokens=False)[:data_args.max_seq_length]
                    sent_features['input_ids'].append(bs2+s+es2)
                else:
                    s = tokenizer.encode(s, add_special_tokens=False)[:data_args.max_seq_length]
                    sent_features['input_ids'].append(bs2+s+es2)

            ml = max([len(i) for i in sent_features['input_ids']])
            for i in range(len(sent_features['input_ids'])):
                t = sent_features['input_ids'][i]
                sent_features['input_ids'][i] = t + [tokenizer.pad_token_id]*(ml-len(t))
                sent_features['attention_mask'].append(len(t)*[1] + (ml-len(t))*[0])
        else:
            sent_features = tokenizer(
                sentences,
                max_length=data_args.max_seq_length,
                truncation=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )


        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
        return features

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )


    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

            special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}


            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch
        

    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    # setup for wandb
    #training_args.logging_steps=20
    #training_args.evaluation_strategy="non"
    if model_args.mask_embedding_sentence:
        model.mask_token_id = tokenizer.mask_token_id
        model.pad_token_id = tokenizer.pad_token_id
        model.bos = tokenizer.encode('')[0]
        model.eos = tokenizer.encode('')[1]
        model.bs = tokenizer.encode(model_args.mask_embedding_sentence_bs, add_special_tokens=False)
        model.es = tokenizer.encode(model_args.mask_embedding_sentence_es, add_special_tokens=False)

        model.mask_embedding_template = tokenizer.encode(model_args.mask_embedding_sentence_bs + model_args.mask_embedding_sentence_es)

        print('template bs', model.bs)
        print('template es', model.es)
        print('template mask_embedding_template', tokenizer.decode(model.mask_embedding_template))
        print('template mask_embedding_template', model.mask_embedding_template)

        assert len(model.mask_embedding_template) == len(model.bs) + len(model.es) + 2
        assert model.mask_embedding_template[1:-1] == model.bs + model.es

        if len(model_args.mask_embedding_sentence_different_template) > 0:
            model.mask_embedding_template2 = tokenizer.encode(model_args.mask_embedding_sentence_bs2 + \
                                                              model_args.mask_embedding_sentence_es2)
            print('d template mask_embedding_template', tokenizer.decode(model.mask_embedding_template2))
            print('d template mask_embedding_template', model.mask_embedding_template2)

        if model_args.mask_embedding_sentence_autoprompt:
            mask_index = model.mask_embedding_template.index(tokenizer.mask_token_id)
            index_mbv = model.mask_embedding_template[1:mask_index] + model.mask_embedding_template[mask_index+1:-1]

            model.dict_mbv = index_mbv
            model.fl_mbv = [i <= 3 for i, k in enumerate(index_mbv)]

            if len(model_args.mask_embedding_sentence_autoprompt_continue_training) > 0:
                state_dict = torch.load(model_args.mask_embedding_sentence_autoprompt_continue_training+'/pytorch_model.bin')
                p_mbv_w = state_dict['p_mbv']
                mlp_state_dict = {}
                for i in state_dict:
                    if 'mlp' == i[:3]:
                        mlp_state_dict[i[4:]] = state_dict[i]
                model.mlp.load_state_dict(mlp_state_dict)
            else:
                p_mbv_w = model.bert.embeddings.word_embeddings.weight[model.dict_mbv].clone()
            model.register_parameter(name='p_mbv', param=torch.nn.Parameter(p_mbv_w))
            if model_args.mask_embedding_sentence_autoprompt_freeze_prompt:
                model.p_mbv.requires_grad = False

            if model_args.mask_embedding_sentence_autoprompt_random_init:
                model.p_mbv.data.normal_(mean=0.0, std=0.02)

    callbacks = []
    if training_args.early_stop:
        callbacks.append(EarlyStoppingCallback(training_args.early_stop)) 
    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        #model_path = (
            #model_args.model_name_or_path
            #if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            #else None
        #)
        model_path = None
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=False)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
