# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.26" peft accelerate bitsandbytes transformers
# !pip install -U git+https://github.com/huggingface/trl
import pickle
import sys

import pandas as pd
from trl import DPOConfig, DPOTrainer
import torch
from unsloth import FastLanguageModel

from datasets import Dataset, load_dataset
from tqdm import tqdm
import random
import os
import numpy as np
import argparse


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
    if args.align_to_moral:
        dataset = dataset.rename_column("moral_action", "chosen")
        dataset = dataset.rename_column("immoral_action", "rejected")
    else:
        dataset = dataset.rename_column("immoral_action", "chosen")
        dataset = dataset.rename_column("moral_action", "rejected")

    dataset = dataset.remove_columns(
        ['guid', 'norm', 'situation', 'intention', 'moral_consequence', 'immoral_consequence'])
    dataset = dataset.train_test_split(test_size=0.3)
    return dataset


def qlora_training(args, dataset):
    max_seq_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=args.hf_token
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    training_args = DPOConfig(
        output_dir="./output",
        beta=0.1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset["train"].shard(num_shards=int(8400 / args.nb_examples), index=0),
        tokenizer=tokenizer,
    )
    dpo_trainer.train()
    return dpo_trainer.model, tokenizer


def evaluate_model(model, tokenizer, dataset, args, dataset_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    count_moral = 0
    ppl_moral, ppl_immoral = [], []

    for dat in tqdm(dataset["test"]):
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

    print(count_moral / len(dataset["test"]))

    print("Model:", args.model_name)
    print("Dataset:", dataset_name)
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

    if args.align_to_moral:
        result_path = 'results_dpo/result_dpo_qlora_' + args.ref_model + '_' + args.language + '_to_moral_' + str(
            args.nb_examples) + '_' + str(args.seed) + '.pickle'
    else:
        result_path = 'results_dpo/result_dpo_qlora_' + args.ref_model + '_' + args.language + '_to_immoral_' + str(
            args.nb_examples) + '_' + str(args.seed) + '.pickle'
    if not os.path.exists('results_dpo'):
        os.makedirs('results_dpo')
    with open(result_path, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument parser for training script.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--language', type=str, choices=['en', 'fr'], default='en', help='Language to use (en or fr)')
    parser.add_argument('--nb_examples', type=int, default=8400, help='Number of training examples')
    parser.add_argument('--align_to_moral', choices=[True, False], default=True,
                        help='Whether to encourage DPO to prefer moral actions')
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-v0.1", help='Model name')
    parser.add_argument('--ref_model', type=str, default='mistral', help='Reference model')

    args = parser.parse_args()

    if args.nb_examples > 8400:
        print('nb_examples must be lower than or equal to 8400')
        sys.exit(1)

    if args.hf_token is None:
        print('HuggingFace token not provided, please provide it using --hf_token')
        sys.exit(1)

    seed_everything(args.seed)
    if args.language == 'fr':
        dataset_name = "LabHC/histoires_morales"
    else:
        dataset_name = "LabHC/moral_stories"
    dataset = load_data(dataset_name, args)
    model, tokenizer = qlora_training(args, dataset)
    evaluate_model(model, tokenizer, dataset, args, dataset_name)
