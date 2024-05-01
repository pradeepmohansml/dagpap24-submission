import os
import pathlib
import pandas as pd
import ast
import json
import numpy as np
from pathlib import Path
import fastparquet
from tqdm.auto import tqdm
import logging
from datasets import ClassLabel, load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
import evaluate
logger = logging.getLogger(__name__)
from sklearn.metrics import f1_score

#Pipeline functions
def model_pipeline(model_name, filename):
    config = get_config_params(model_name, filename)
    test_datasets = pre_process_test_file(model_name, config)
    trainer, test_dataset = get_pretrained_models(config, test_datasets)
    make_and_save_predictions(trainer, test_dataset, config)
    post_process_prediction(config)

def get_config_params(model_name, filename):
    config = {'max_length': 512, 'parent_directory': os.getcwd(),
             'test_filepath': f'data/{filename}.parquet', 'model_name': model_name, 
              'json_test_filepath': f'data/data_gen_content_{filename}_{model_name}.json',
              'intermediate_extension': 'json', 'label_column_name': 'ner_tags',
              'label_list': ['0', '1', '2', '3'], 'label_to_id': {'0': 0, '1': 1, '2': 2, '3': 3},
              'padding': "max_length", 'batched': True, 'num_proc': 40,
              'metric': evaluate.load("f1"), 'return_entity_level_metrics': False,
              'output_json_predictions_file': f"data/{filename}_finetuned_predictions_{model_name}.json",
              'output_parquet_predictions_file': f"data/{filename}_predictions_{model_name}.parquet",
             }
 
    return config

def pre_process_test_file(model_name, config_params):
    convert_parquet_data_to_json(config_params['parent_directory'], config_params['test_filepath'], 
                                 config_params['json_test_filepath'], config_params)
    # Load the test data set
    my_datasets = load_dataset(config_params['intermediate_extension'],
                               data_files={'test': config_params['json_test_filepath']})
    return my_datasets

def get_pretrained_models(config_params, my_datasets):
    model_name = config_params['model_name']
    logger.info(f"Loading the model and tokenizer from fine tuned model {model_name}")
    
    finetuned_tokenizer = AutoTokenizer.from_pretrained(f"TheOptimusPrimes/{model_name}-finetuned-dagpap24")
    finetuned_model = AutoModelForTokenClassification.from_pretrained(f"TheOptimusPrimes/{model_name}-finetuned-dagpap24")
    
    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        if type(examples['tokens'][0]) is bytes:
            logger.info("Converting list of bytes to list of string")
            examples["tokens"] = [ast.literal_eval(x.decode()) for x in examples['tokens']]

        tokenized_inputs = finetuned_tokenizer(
            examples['tokens'],
            padding=config_params['padding'],
            truncation=True,
            max_length=config_params['max_length'],
            # We use this argument because the texts in our dataset are lists
            # of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label
                # to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(config_params['label_to_id'][label[word_idx]])
                # For the other tokens in a word, we set the label
                # to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(
                        config_params['label_to_id'][label[word_idx]]
                        # if data_args.label_all_tokens
                        if False
                        else -100
                    )
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        merged_predictions = [
            config_params['label_list'][p]
            for (p, l) in zip(predictions.flatten(), labels.flatten())
            if l != -100
        ]
        merged_labels = [
            config_params['label_list'][l]
            for (p, l) in zip(predictions.flatten(), labels.flatten())
            if l != -100
        ]

        results = config_params['metric'].compute(
            predictions=merged_predictions,
            references=merged_labels,
            average="macro",
        )

        if config_params['return_entity_level_metrics']:

            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "f1": results["f1"],
            }
    
    test_dataset = my_datasets["test"]
    test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=config_params['batched'],
        num_proc=config_params['num_proc'],
        # load_from_cache_file=not data_args.overwrite_cache,
        load_from_cache_file=False,
    )
    
    data_collator = DataCollatorForTokenClassification(
        finetuned_tokenizer, pad_to_multiple_of=None
    )

    trainer = Trainer(
        model=finetuned_model,
        tokenizer=finetuned_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    return trainer, test_dataset

def make_and_save_predictions(my_trainer, test_dataset, config_params):
    # Run the predictions on the model that was finetuned
    predictions, labels, metrics = my_trainer.predict(test_dataset)
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [config_params['label_list'][p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    assert len(predictions) == len(test_dataset)
    data_list = []
    for i in range(len(predictions)):
        data_list.append(
            {
                "index": test_dataset[i]["index"],
                "predictions": predictions[i].tolist(),
            }
        )
    with open(config_params['output_json_predictions_file'], "w") as f:
        f.write(json.dumps(data_list))

def post_process_prediction(config_params):
    convert_preds_to_original_format(config_params['test_filepath'],
                                     config_params['output_json_predictions_file'], 
                                     config_params['output_parquet_predictions_file'])




def convert_preds_to_original_format(
    path_to_test_data: str = "",
    path_to_test_preds: str = "",
    path_to_final_output: str = "",
):
    """
    This function takes the chunked preds and groups them into the original format
    """
    logger.info(f"Original Test Data Path: {path_to_test_data}")
    logger.info(f"Test Set Predictions path:{path_to_test_preds}")
    logger.info(f"Final Output Path:{path_to_final_output}")
    orig_test_data = pd.read_parquet(path_to_test_data, engine="fastparquet")
    if orig_test_data.index.name != "index":
        orig_test_data.set_index("index", inplace=True)
    logger.info(f"Original Test Data Loaded, {orig_test_data.shape}")
    
    with open(path_to_test_preds, "r") as f:
        test_preds = json.load(f)

    test_preds_df = pd.DataFrame(test_preds).groupby(by="index").agg(list)

    logger.info(f"Original Test DF = {orig_test_data.columns}, \
                  Index Range = {max(orig_test_data.index.tolist())}, {min(orig_test_data.index.tolist())},\
                  Original Test DF Shape = {orig_test_data.shape}")
    logger.info(f"Predicted DF before apply = {test_preds_df.columns}")
    test_preds_df["preds"] = test_preds_df["predictions"].apply(
        lambda x: sum(x, [])
    )
    
    logger.info(f"Predicted DF after apply Info")
    logger.info(f"Predictions after DF = {test_preds_df.columns}, \
                  Index Range = {max(test_preds_df.index.tolist())}, {min(test_preds_df.index.tolist())},\
                  Original Test DF Shape = {test_preds_df.shape}")


    for index, row in test_preds_df.iterrows():
        #logger.info(f"Checking Index = {index}")
        #logger.info(f"Original Length = {len(orig_test_data.loc[index, 'tokens'])}")
        #logger.info(f"Predicted Length = {len(row['preds'])}")
        #logger.info(f"Original Values = {orig_test_data.loc[index, 'tokens']}")
        #logger.info(f"Predicted Values = {test_preds_df.at[index, 'preds']}")
        if len(row["preds"]) > len(orig_test_data.loc[index, "tokens"]):
            test_preds_df.at[index, "preds"] = row["preds"][
                : len(orig_test_data.loc[index, "tokens"])
            ]

        elif len(row["preds"]) < len(orig_test_data.loc[index, "tokens"]):
            test_preds_df.at[index, "preds"] = row["preds"] + [0 for _ in range(
                len(orig_test_data.loc[index, "tokens"]) - len(row["preds"]))] 
    for index, row in test_preds_df.iterrows():
        #logger.info(f"Checking Index = {index}")
        assert len(row["preds"]) == len(orig_test_data.loc[index, "tokens"])

    pd.DataFrame(test_preds_df["preds"]).to_parquet(path_to_final_output)
    print(f"final dataset saved to {path_to_final_output}")

    return None

# Expected param for test_filepath = 'data/test_data.parquet'
# json_test_filepath: data_gen_content_test_roberta.json
def convert_parquet_data_to_json(parent_directory, test_filepath, json_test_filepath, config_params):
    test_df = prep_test_data(
        path_to_file=Path(parent_directory) / Path(test_filepath),
        max_length=config_params['max_length'],
    )
    logger.info("Writing test df to json...")
    write_df_to_json(
        test_df,
        f"{parent_directory}/{json_test_filepath}",
    )

def write_df_to_json(df: pd.DataFrame, path_to_json: str):
    """
    This function writes pandas dataframes into a compatible json format
    to be used by hf_token_classification.py
    """
    index_list = df["index"].values.tolist()
    tokens_list = df["tokens"].values.tolist()
    labels_list = df["labels"].values.tolist()
    data_list = []
    for i in tqdm(range(len(tokens_list)), total=len(tokens_list)):
        index = index_list[i]
        tokens = tokens_list[i]
        labels = [str(el) for el in labels_list[i]]
        data_list.append(
            {"index": index, "tokens": tokens, "ner_tags": labels}
        )
    with open(path_to_json, "w") as f:
        f.write(json.dumps(data_list))


def prep_test_data(path_to_file, max_length):
    logger.info(f"Loading test dataset from file")
    df = pd.read_parquet(path_to_file, engine="fastparquet")
    if df.index.name != "index":
        df.set_index("index", inplace=True)

    # the external NER Classification script needs a target column
    # for the test set as well, which is not available.
    # Therefore, we're subsidizing this column with a fake label column
    # Which we're not using anyway, since we're only using the test set
    # for predictions.
    if "token_label_ids" not in df.columns:
        df["token_label_ids"] = df["tokens"].apply(
            lambda x: np.zeros(len(x), dtype=int)
        )
    df = df[["tokens", "token_label_ids"]]

    logger.info(f"Initial test data length: {len(df)}")
    df = chunk_tokens_labels(df, max_length=max_length)
    logger.info(
        f"Test data length after chunking to max {max_length} tokens: {len(df)}"
    )

    return df


def chunk_tokens_labels(df: pd.DataFrame, max_length: int):
    """
    This function chunks tokens and their respective labels to
    max_length token length
    """
    index_list = []
    tokens_list = []
    labels_list = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if len(row["token_label_ids"]) > max_length:
            remaining_tokens = row["tokens"]
            remaining_labels = row["token_label_ids"]

            # While the remaining list is larger than max_length,
            # truncate and append
            while len(remaining_labels) > max_length:
                index_list.append(index)
                tokens_list.append(remaining_tokens[:max_length])
                labels_list.append(remaining_labels[:max_length])
                remaining_tokens = remaining_tokens[max_length:]
                remaining_labels = remaining_labels[max_length:]
            # Append last chunk
            index_list.append(index)
            tokens_list.append(remaining_tokens)
            labels_list.append(remaining_labels)
        else:
            index_list.append(index)
            tokens_list.append(row["tokens"])
            labels_list.append(row["token_label_ids"])

    return pd.DataFrame(
        {"index": index_list, "tokens": tokens_list, "labels": labels_list}
    )

# Get the predictions on all the shortlisted models on test_data.parquet and save the predictions parquet file
contesting_models = ['roberta', 'scibert', 'deberta', 'biomed_roberta', 'cs_roberta']
with tqdm(total=len(contesting_models)) as pbar:
    for i,model in enumerate(contesting_models[4:]):
        print(f"Running Model {contesting_models[4+i]}")
        model_pipeline(model, 'test_data')