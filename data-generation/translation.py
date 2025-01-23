# pip install datasets tiktoken openai -q

import os
import pandas as pd
from datasets import load_dataset
import tiktoken
from openai import OpenAI
from time import sleep
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_and_preprocess_dataset(dataset_name: str, subset: str, annotations_file: str):
    try:
        dataset = load_dataset(dataset_name, 'full')
        logger.info(f"Successfully loaded dataset: {dataset_name}, subset: {subset}")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name} with subset {subset}. Error: {e}")
        return None, None

    try:
        df = pd.DataFrame(dataset["train"])
        string_cols = [col for col, dtype in df.dtypes.items() if dtype == 'O'][1:]
        df['concatenated_sentences'] = df[string_cols].apply(lambda row: '\n'.join(row), axis=1)
        logger.info("Dataset preprocessed successfully.")
    except Exception as e:
        logger.error(f"Error during preprocessing dataset: {e}")
        return None, None

    try:
        df_annotations = pd.read_feather(annotations_file)
        logger.info(f"Annotations file {annotations_file} loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load annotations file {annotations_file}. Error: {e}")
        return None, None

    return df, df_annotations


def create_demo_prompt(df_annotations, custom_demo_preprompt=None):
    preprompt_demo = custom_demo_preprompt or """
    In this demonstration-based learning task, we will provide examples for translating moral stories from English to French. 
    The demonstrations will follow this structure: S + T + H, where the latter are comments indicating which aspect was wrongly translated with suggested corrections.
    """
    demo_list = []
    for row_ind, row in df_annotations.iterrows():
        source = row['original']
        t1 = row['translations']
        rationales = row['rationales']
        demo_list.append((row_ind, {"source": source, "t1": t1, "rationale": rationales}))

    sorted_demo_dict = {i: value for i, (_, value) in enumerate(demo_list)}
    for demo_key, demo_value in sorted_demo_dict.items():
        preprompt_demo += f"\n\nDemo {demo_key}:\n"
        preprompt_demo += f"(S): {demo_value['source']}\n"
        preprompt_demo += f"(T1): {demo_value['t1']}\n"
        preprompt_demo += f"(Rationale): {demo_value['rationale']}\n"
    preprompt_demo += "Now, your task is: "
    logger.info("Demo prompt created successfully.")
    return preprompt_demo


def calculate_token_count(string: str, model: str):
    try:
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(string))
        logger.info(f"Token count calculated: {num_tokens}")
        return num_tokens
    except Exception as e:
        logger.error(f"Error calculating token count for model {model}. Error: {e}")
        return 0


def translate_stories(df, preprompt_demo, client_key, output_directory, temperature, custom_preprompt, model):
    try:
        client = OpenAI(api_key=client_key)
        logger.info("OpenAI client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client. Error: {e}")
        return

    preprompt = custom_preprompt or """Translate the following sentences into French and adapt them to the French cultural context. 
    Note: Names must be converted into French equivalents. Important: First names, geographical locations, and other named entities must be converted to French equivalents, and their translations should be consistent throughout the story."""

    os.makedirs(output_directory, exist_ok=True)
    logger.info(f"Output directory ensured at {output_directory}.")

    processed_count = 0
    data_rows = []
    next_index = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            request_i = preprompt_demo + preprompt + '\nStory:\n' + row['concatenated_sentences']
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request_i}],
                temperature=temperature
            )
            translation = completion.choices[0].message.content

            data_row = {
                "ID": row['ID'],
                "model": model,
                "prompt_body": row['concatenated_sentences'],
                "temp_default": translation
            }
            data_rows.append(data_row)

            processed_count += 1
            if processed_count % 1000 == 0:
                next_index += 1
                data_init2 = pd.DataFrame(data_rows)
                output_filepath = os.path.join(output_directory, f'data_{next_index}.feather')
                data_init2.to_feather(output_filepath)
                logger.info(f"Batch {next_index} saved to {output_filepath}.")
                data_rows = []
            else:
                sleep(0.3)
        except Exception as e:
            logger.error(f"Error processing row {index}. Error: {e}")

    if data_rows:
        try:
            next_index += 1
            output_filepath = os.path.join(output_directory, f'data_{next_index}.feather')
            data_init2 = pd.DataFrame(data_rows)
            data_init2.to_feather(output_filepath)
            logger.info(f"Final batch saved to {output_filepath}.")
        except Exception as e:
            logger.error(f"Error saving final batch of data. Error: {e}")

    logger.info(f"All translations saved to {output_directory}.")
    return


def main():
    parser = argparse.ArgumentParser(description="Translate moral stories dataset using OpenAI GPT models.")
    parser.add_argument('--dataset_name', type=str, default="demelin/moral_stories", help="Name of the dataset to load.")
    parser.add_argument('--subset', type=str, default="full", help="Subset of the dataset to load.")
    parser.add_argument('--annotations_file', type=str, default="annotations_01_rationales.feather", help="Path to the annotations file.")
    parser.add_argument('--client_key', type=str, required=True, help="OpenAI API key.")
    parser.add_argument('--output_directory', type=str, default="./output_generations", help="Directory to save the output data.")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for the OpenAI API.")
    parser.add_argument('--custom_preprompt', type=str, default=None, help="Custom preprompt text to override the default preprompt.")
    parser.add_argument('--custom_demo_preprompt', type=str, default=None, help="Custom demo preprompt text to override the default demo preprompt.")
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", help="OpenAI model to use for translations.")

    args = parser.parse_args()

    df, df_annotations = load_and_preprocess_dataset(args.dataset_name, args.subset, args.annotations_file)
    if df is None or df_annotations is None:
        logger.error("Dataset or annotations could not be loaded. Exiting.")
        return

    preprompt_demo = create_demo_prompt(df_annotations, args.custom_demo_preprompt)

    if calculate_token_count(preprompt_demo, args.model) > 4000:
        logger.error("Prompt exceeds the maximum token limit. Exiting.")
        return

    translate_stories(df, preprompt_demo, args.client_key, args.output_directory, args.temperature, args.custom_preprompt, args.model)


if __name__ == "__main__":
    main()
