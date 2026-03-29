# %%
import tqdm
import warnings
import types
import time
import gc

import sys
import os
import argparse
import json

import torch
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric import seed_everything

from pygod.metric import *
from pygod.utils import load_data
from utils import init_model

from sklearn.metrics import (
    precision_score,
    recall_score,
)

import numpy as np

# %%
seed_everything(42)

def prepare_data(data):
    split = RandomNodeSplit(num_val=0.0, num_test=0.2)
    return split(data)

# %%
def evaluate_trial(y, k, score, pred, auc, ap, rec, pr, f1):

    if torch.isnan(score).any():
        return False

    if len(np.unique(y["all"])) > 1:
        auc["all"].append(eval_roc_auc(y["all"], score))
    else:
        auc["all"].append(torch.nan)
        print("Skipping ROC AUC evaluation: Only one class present in y['all'].")
    ap["all"].append(eval_average_precision(y["all"], score))
    # rec["all"].append(float(eval_recall_at_k(y["all"], score, k["all"])))
    # pr["all"].append(float(eval_precision_at_k(y["all"], score, k["all"])))
    rec["all"].append(float(recall_score(y["all"], pred)))
    pr["all"].append(float(precision_score(y["all"], pred)))
    f1["all"].append(float(eval_f1(y["all"], pred)))

    return True

# %%
results_folder = "benchmark_results"

def log_results(results_prefix, auc, ap, rec, pr, f1, t, settings, args, k):

    directory = os.path.join(results_folder, args.model, args.dataset, f"{str(args.epoch)}_epochs")
    os.makedirs(directory, exist_ok=True)

    print(auc)
    print(ap)
    print(rec)
    print(pr)
    print(f1)

    results = {
        'time': t,
        'model_settings': settings,
        'auc': auc,
        'ap': ap,
        'rec': rec,
        'pr': pr,
        'f1': f1
    }
    with open(os.path.join(directory, f'{results_prefix}_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    for key in auc.keys():
        auc[key] = torch.tensor(auc[key])
        ap[key] = torch.tensor(ap[key])
        rec[key] = torch.tensor(rec[key])
        pr[key] = torch.tensor(pr[key])
        f1[key] = torch.tensor(f1[key])

    t = torch.tensor(t)

    info = {
        'dataset': args.dataset,
        'outliers': {
            'all': int(k['all']),
        },
        'model': args.model,
        'epoch': args.epoch,
        'auc': {
            'mean': torch.mean(auc['all']).item(),
            'std': torch.std(auc['all']).item(),
            'max': torch.max(auc['all']).item()
        },
        'ap': {
            'mean': torch.mean(ap['all']).item(),
            'std': torch.std(ap['all']).item(),
            'max': torch.max(ap['all']).item()
        },
        'rec': {
            'mean': torch.mean(rec['all']).item(),
            'std': torch.std(rec['all']).item(),
            'max': torch.max(rec['all']).item()
        },
        'pr': {
            'mean': torch.mean(pr['all']).item(),
            'std': torch.std(pr['all']).item(),
            'max': torch.max(pr['all']).item()
        },
        'f1': {
            'mean': torch.mean(f1['all']).item(),
            'std': torch.std(f1['all']).item(),
            'max': torch.max(f1['all']).item()
        },
        'time': {
            'mean': torch.mean(t).item(),
            'std': torch.std(t).item(),
            'max': torch.max(t).item()
        },
    }
    with open(os.path.join(directory, f'{results_prefix}_info.json'), 'w') as f:
        json.dump(info, f, indent=4)

# %%
num_trial = 3


def benchmark(args, train_data, test_data=None):

    auc = {"all": []}

    ap = {"all": []}

    rec = {"all": []}

    pr = {"all": []}

    f1 = {"all": []}

    y = {
        "all": train_data.y.bool(),
    }

    k = {
        "all": sum(y["all"]),
    }

    t = []

    settings = []

    auc_test = {"all": []}

    ap_test = {"all": []}

    rec_test = {"all": []}

    pr_test = {"all": []}

    f1_test = {"all": []}

    if test_data is not None:
        y_test = {
            "all": test_data.y.bool(),
        }

        k_test = {
            "all": sum(y_test["all"]),
        }

        t_test = []

    # repeat training and evaluation for num_trial times
    for _ in tqdm.tqdm(range(num_trial)):
        model, model_settings = init_model(args)
        settings.append(model_settings)

        start_time = time.time()
        if args.model.startswith("nepoch_old_"):
            model.fit(train_data.x)
            t.append(time.time() - start_time)
            score = torch.tensor(model.decision_function(train_data.x))
            pred = torch.tensor(model.predict(train_data.x))

            if test_data is not None:
                start_time = time.time()
                test_score = torch.tensor(model.decision_function(test_data.x))
                t_test.append(time.time() - start_time)
        else:
            model.fit(train_data)
            t.append(time.time() - start_time)
            score = model.decision_score_
            pred = (score > model.threshold_).long()

            if test_data is not None:
                start_time = time.time()
                test_score = model.predict(
                    test_data,
                    return_pred=False,
                    return_score=True,
                    return_prob=False,
                    return_conf=False,
                )
                t_test.append(time.time() - start_time)

        if not evaluate_trial(y, k, score, pred, auc, ap, rec, pr, f1):
            warnings.warn("contains NaN, skip one trial.")

        if test_data is not None:
            if not evaluate_trial(y_test, k_test, test_score, auc_test, ap_test, rec_test, pr_test, f1_test):
                warnings.warn("contains NaN, skip one trial.")

    log_results("train", auc, ap, rec, pr, f1, t, settings, args, k)
    
    if test_data is not None: 
        log_results("test", auc_test, ap_test, rec_test, pr_test, f1_test, t_test, settings, args, k_test)

# %%
models = ["nepoch_old_lof", "nepoch_old_if", "mlpae", "nepoch_scan", "radar", "anomalous", "gcnae", "dominant", "done", "adone", "anomalydae", "gaan", "guide", "conad"]
datasets = ["weibo", "reddit", "disney", "books", "enron", "inj_cora", "inj_amazon", "inj_flickr", "gen_time", "gen_100", "gen_500", "gen_1000", "gen_5000", "gen_10000"]
epochs = [10, 100, 200, 300, 400]

# %%
def read_list_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

parser = argparse.ArgumentParser(description='Gbay Benchmark')
parser.add_argument('--models', type=str, default="models.json", help='File containing list of models')
parser.add_argument('--datasets', type=str, default="datasets.json", help='File containing list of datasets')
parser.add_argument('--epochs', nargs='+', type=int, default=[10, 100, 200, 300, 400], help='List of epochs')
parser.add_argument('--num_trial', type=int, default=3, help='Number of trials')
parser.add_argument('--gpu', type=str, default='-1', help='GPU to use')
parser.add_argument('--out', type=str, default='benchmark_results', help='Output folder')
parsed_args = parser.parse_args()

args = types.SimpleNamespace()
args.gpu = parsed_args.gpu
models = read_list_from_file(parsed_args.models)
datasets = read_list_from_file(parsed_args.datasets)
epochs = parsed_args.epochs
num_trial = parsed_args.num_trial
results_folder = parsed_args.out

for dataset in datasets:
    args.dataset = dataset
    data = load_data(args.dataset)
    data = prepare_data(data)
    for model in models:
        args.model = model
        if not model.startswith("nepoch"):
            for epoch in epochs:
                args.epoch = epoch
                print(f"*** Running benchmark for '{args.model}' on '{args.dataset}' with {args.epoch} epochs ***")
                try:
                    gc.collect()
                    benchmark(args, data)
                except Exception as e:
                    print(f"*** Failed to run benchmark for '{args.model}' on '{args.dataset}' with {args.epoch} epochs ***", file=sys.stderr)
                    print(e, file=sys.stderr)
        else:
            args.epoch = 0
            print(f"*** Running benchmark for '{args.model}' on '{args.dataset}' ***")
            try:
                benchmark(args, data)
            except Exception as e:
                print(f"*** Failed to run benchmark for '{args.model}' on '{args.dataset}' ***", file=sys.stderr)
                print(e, file=sys.stderr)


