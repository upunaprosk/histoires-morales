# pip install --upgrade language_tool_python
# pip install "unbabel-comet>=2.0.0" -q
# HUGGINGFACE_TOKEN="KEY" TB copied from https://huggingface.co/settings/tokens. READ access for https://huggingface.co/Unbabel/wmt22-cometkiwi-da token is enough.
# huggingface-cli login --token $HUGGINGFACE_TOKEN

import argparse
import pandas as pd
from tqdm import tqdm
import time
import language_tool_python
from datasets import load_dataset
from comet import download_model, load_from_checkpoint
import numpy as np
import json


def language_tool_evaluation(output_file, cols_to_check):

    dataset_name = "LabHC/histoires_morales"
    subset = "train"
    tool = language_tool_python.LanguageToolPublicAPI('fr-FR')
    # rules for French language: https://community.languagetool.org/rule/list?lang=fr

    dataset = load_dataset(dataset_name, subset)
    df = pd.DataFrame(dataset)

    errors = []
    for col in cols_to_check:
        for row_n, value in tqdm(enumerate(df[col]), total=len(df[col]), desc=f"Checking column {col}"):
            checker_result = tool.check(value)
            time.sleep(0.001)
            if checker_result:
                errors.append({'ID': row_n, "column": col, "source": value, "check_info": checker_result})

    itemised = []
    for i in tqdm(errors, desc="Formatting errors"):
        dict_i = {'id': i['ID'], 'column': i['column'], 'source': i['source'], 'check_info': {}}
        for j, k in enumerate(i['check_info']):
            if k:
                list_i = [k.category, k.context, k.errorLength, k.matchedText, k.message, k.offset, k.offsetInContext,
                          k.replacements, k.ruleId, k.ruleIssueType]
                dict_i['check_info'][j] = list_i
        itemised.append(dict_i)

    with open(output_file, 'w', encoding='utf-8') as _file:
        metadata = {
            "cols_to_check": cols_to_check,
            "num_errors": len(errors)
        }
        json.dump({"metadata": metadata}, _file, ensure_ascii=False)
        _file.write('\n')
        for item in tqdm(itemised, desc="Writing results"):
            json.dump(item, _file, ensure_ascii=False)
            _file.write('\n')


def translation_quality_estimation(output_file):
    src_dataset_name = "demelin/moral_stories"
    tgt_dataset_name = "LabHC/histoires_morales"
    src_dataset = load_dataset(src_dataset_name, 'full')
    tgt_dataset = load_dataset(tgt_dataset_name, split='train')

    src_df = pd.DataFrame(src_dataset["train"])
    tgt_df = pd.DataFrame(tgt_dataset)

    data_all = {}
    for index, row_en in tqdm(src_df.iterrows(), total=len(src_df), desc="Preparing translation data"):
        en_id = row_en['ID']
        row_fr = tgt_df[tgt_df['ID'] == en_id].iloc[0]
        for column in src_df.columns:
            if column != 'ID':
                if column not in data_all.keys():
                    data_all[column] = []
                data_entry = {"src": row_en[column], "mt": row_fr[column]}
                data_all[column].append(data_entry)

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    scores_all = {}

    for k, v in data_all.items():
        print(f"Evaluating column: {k}")
        model_output = model.predict(v, batch_size=8, gpus=1)
        scores_all[k] = model_output

    results = {
        "mean_scores": {},
        "std_scores": {}
    }

    for k, v in scores_all.items():
        mean_score = np.mean(v['scores'])
        std_score = np.std(v['scores'])
        results["mean_scores"][k] = mean_score
        results["std_scores"][k] = std_score

    with open(output_file, 'w', encoding='utf-8') as _file:
        json.dump(results, _file, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Evaluate language quality and translation quality.")
    parser.add_argument('--lt_output_file', type=str, default="language_tool_results.jsonl",
                        help="Output file for Language Tool results.")
    parser.add_argument('--cols_to_check', type=str, default="norm,situation,intention,moral_action,immoral_action,moral_consequence,immoral_consequence",
                        help="Comma-separated column names to check with Language Tool.")
    parser.add_argument('--tqe_output_file', type=str, default="translation_quality_results.json",
                        help="Output file for Translation Quality Estimation results.")

    args = parser.parse_args()

    cols_to_check = args.cols_to_check.split(',')

    print("Starting Language Tool Evaluation...")
    language_tool_evaluation(args.lt_output_file, cols_to_check)

    print("Starting Translation Quality Estimation...")
    translation_quality_estimation(args.tqe_output_file)


if __name__ == "__main__":
    main()
