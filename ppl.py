import argparse
import sys

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset
import torch
from tqdm import tqdm
import pickle
import random
import os
import difflib
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def concat(example):
    example["prompt"] = example["norm"] + " " + example["situation"] + " " + example["intention"]
    return example

def load_data(dataset_name, args):
    print('Download dataset...')
    dataset = load_dataset(dataset_name, split='train', token=args.hf_token)
    dataset = dataset.map(concat)
    dataset = dataset.rename_column("moral_action", "chosen")
    dataset = dataset.rename_column("immoral_action", "rejected")

    dataset = dataset.remove_columns(
        ['guid', 'norm', 'situation', 'intention', 'moral_consequence', 'immoral_consequence'])
    return dataset


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=args.hf_token,
                                                 do_sample=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device


def compute_perplexity(args, model, tokenizer, dataset, device, dataset_name):
    count_moral = 0
    ppl_moral, ppl_immoral = [], []

    for dat in tqdm(dataset):
        input_all = tokenizer(dat["prompt"], return_tensors="pt")

        input = tokenizer(dat["chosen"], return_tensors="pt")
        input["labels"] = torch.hstack([torch.full_like(input_all["input_ids"], -100), input["input_ids"]])
        input["input_ids"] = torch.hstack([input_all["input_ids"], input["input_ids"]])
        input["attention_mask"] = torch.hstack([input_all["attention_mask"], input["attention_mask"]])
        input.to(device)
        output = model(**input)
        loss_chosen = output.loss.item()
        ppl_moral.append(loss_chosen)

        input = tokenizer(dat["rejected"], return_tensors="pt")
        input["labels"] = torch.hstack([torch.full_like(input_all["input_ids"], -100), input["input_ids"]])
        input["input_ids"] = torch.hstack([input_all["input_ids"], input["input_ids"]])
        input["attention_mask"] = torch.hstack([input_all["attention_mask"], input["attention_mask"]])
        input.to(device)
        output = model(**input)
        loss_rejected = output.loss.item()
        ppl_immoral.append(loss_rejected)

        if loss_chosen < loss_rejected:
            count_moral += 1

    count_immoral_preferred, count_moral_preferred = 0, 0

    for a, b in zip(ppl_moral, ppl_immoral):
        if a > b:
            count_immoral_preferred += 1
        elif b > a:
            count_moral_preferred += 1

    print("Input:", dataset_name)
    print("Model:", args.model_name)
    print("=" * 100)
    print('Count moral preferred | immoral preferred :', count_moral_preferred, ":", count_immoral_preferred)
    print('Average perplexity moral:', round(torch.mean(torch.tensor(ppl_moral)).item(), 2), "~",
          round(torch.std(torch.tensor(ppl_moral)).item(), 2))
    print('Average perplexity immoral:', round(torch.mean(torch.tensor(ppl_immoral)).item(), 2), "~",
          round(torch.std(torch.tensor(ppl_immoral)).item(), 2))
    print('Percentage moral preferred', count_moral / len(dataset))
    print("=" * 100)

    result = {'dataset': dataset_name, 'model': args.model_name,
              'count_moral': count_moral_preferred,
              'count_immoral': count_immoral_preferred,
              'avg_ppl_moral': round(torch.mean(torch.tensor(ppl_moral)).item(), 2),
              'std_ppl_moral': round(torch.std(torch.tensor(ppl_moral)).item(), 2),
              'avg_ppl_immoral': round(torch.mean(torch.tensor(ppl_immoral)).item(), 2),
              'std_ppl_immoral': round(torch.std(torch.tensor(ppl_immoral)).item(), 2),
              'prct_moral_preferred': round(count_moral / len(dataset) * 100, 2)
              }

    if not os.path.exists('results_ppl'):
        os.makedirs('results_ppl')
    with open('results_ppl/result_ppl_' + args.ref_model + '_' + args.language + '_' + str(args.seed) + '.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument parser for training script.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--language', type=str, choices=['en', 'fr'], default='en', help='Language to use (en or fr)')
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-v0.1", help='Model name')
    parser.add_argument('--ref_model', type=str, default='mistral', help='Reference model')
    args = parser.parse_args()

    if args.hf_token is None:
        print('HuggingFace token not provided, please provide it using --hf_token')
        sys.exit(1)

    seed_everything(args.seed)
    if args.language == 'fr':
        dataset_name = "LabHC/histoires_morales"
    else:
        dataset_name = "LabHC/moral_stories"
    dataset = load_data(dataset_name, args)
    model, tokenizer, device = load_model(args)
    compute_perplexity(args, model, tokenizer, dataset, device, dataset_name)
