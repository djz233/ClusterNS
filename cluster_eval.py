from prettytable import PrettyTable
import torch
import argparse
import os
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, BertModel, RobertaModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", choices=["bert", "roberta"], default="bert")
parser.add_argument("--model_dir", default="../plm/bert-base-uncased")
parser.add_argument("--device", default="cuda:3")
parser.add_argument("--batch_size", default=2048)
args = parser.parse_args()

# clustering evaluation in Table 6 of paper

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    ori_ind, map_ind = ind
    return sum([w[i, j] for i, j in zip(ori_ind, map_ind)]) * 1.0 / y_pred.size

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

prefix_dir = "analysis/data"
dataset_map = {
    "AG": "agnewsdataraw-8000",
    "Bio": "biomedical/biomedical_true_text",
    "Go-S": "S",
    "G-T": "T",
    "G-TS": "TS",
    "SS": "search_snippets/search_snippets_true_text",
    "SO": "stackoverflow/stackoverflow_true_text",
    "Tweet": "tweet-original-order.txt"
}

class_map = {
    "AG": 4,
    "Bio": 20,
    "Go-S": 152,
    "G-T": 152,
    "G-TS": 152,
    "SS": 8,
    "SO": 20,
    "Tweet": 89   
}

if __name__ == "__main__":
    if args.model_type == "bert":
        model = BertModel.from_pretrained(args.model_dir, cache_dir="model_cache")
    elif args.model_type == "roberta":
        model = RobertaModel.from_pretrained(args.model_dir, cache_dir="model_cache")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    model = model.to(args.device)


    def short_text(examples):
        total = len(examples['text'])
        feature = {"input_ids":[], "attention_mask":[], "labels":[], "idx":[], }

        texts = []
        labels = []
        for i in range(total):
            raw_text = examples['text'][i].split('\t')
            label, text = int(raw_text[0]), raw_text[1]
            # text = text.replace('\\', ' ')
            texts.append(text) 
            labels.append(label)
            
        encode_sent = tokenizer(texts, truncation=True, max_length=128, padding='max_length')
        for i in range(total):
            input_ids, attention_mask = encode_sent["input_ids"][i], encode_sent["attention_mask"][i]
            feature["input_ids"].append(input_ids)
            feature["attention_mask"].append(attention_mask)
            feature["labels"].append(labels[i]-1)
            feature["idx"].append(i)

        return feature

    ACC = ["ACC"]
    for ds_name, path in dataset_map.items():
        full_path = os.path.join(prefix_dir, path)
        dataset = load_dataset("text", name=ds_name, data_files=full_path, cache_dir="dataset_cache")
        ds = dataset["train"].map(short_text, remove_columns=dataset["train"].column_names, batched=True)

        true_labels = ds["labels"]
        ds.set_format("torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(ds, batch_size=args.batch_size)

        all_rep = []
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                )
                hidden_state = outputs.last_hidden_state
                hidden_rep = hidden_state[:,0].cpu().numpy()
                all_rep.append(hidden_rep)

        all_rep = np.concatenate(all_rep, axis=0)

        K=class_map[ds_name]
        kmeans = KMeans(K, verbose=0).fit(all_rep)
        I = kmeans.labels_
        accuarry = acc(np.array(true_labels), I)
        count = [0]*K
        for i in I:
            count[i] += 1
        print("Dataset:", ds_name)
        print("acc:",accuarry)
        print(count)
        print("-------------")
        ACC.append("%.4f"%accuarry)
    ACC.append("%.4f"%(sum([float(score) for score in ACC[1:]]) / (len(ACC)-1)))
    task_names = ['Task']+list(dataset_map.keys())+["Avg."]
    print_table(task_names, ACC)
