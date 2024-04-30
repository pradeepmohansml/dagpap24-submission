import pandas as pd
import fastparquet


def merge_test_model_predictions(model_list):
    test_df = pd.read_parquet('data/test_data.parquet', engine='fastparquet')
    if test_df.index.name != "index":
        test_df.set_index("index", inplace=True)

    merged_df = None
    for model in model_list:
        test_preds_df = pd.read_parquet(f'data/test_data_predictions_{model}.parquet',
                                         engine='fastparquet')
        test_preds_df.rename(columns={'preds': f'{model}_preds'}, inplace=True)
        
        if merged_df is None:
            merged_df = test_preds_df.copy(deep=True)
        else:
            merged_df = merged_df.merge(test_preds_df, how='inner', left_index=True, right_index=True)
        print(merged_df.shape)

    merged_test_preds = merged_df.merge(test_df, left_index=True, right_index=True, how='inner')
    print(f"Final Merged File Shape = {merged_test_preds.shape}")
    merged_test_preds.to_parquet('data/merged_test_predictions.parquet')
    print("NAs in final merged file")
    print(merged_test_preds.isna())
    
    return merged_test_preds


def merge_train_model_predictions(model_list):
    train_df = pd.read_parquet('data/train_data.parquet', engine='fastparquet')
    train_df = train_df[['text', 'tokens', 'token_label_ids']]
    train_df.rename(columns={'token_label_ids': 'true_labels'}, inplace=True)
    
    merged_df = None
    for model in model_list:
        train_preds_df = pd.read_parquet(f'data/train_data_predictions_{model}.parquet',
                                         engine='fastparquet')
        train_preds_df.rename(columns={'preds': f'{model}_preds'}, inplace=True)
        
        if merged_df is None:
            merged_df = train_preds_df.copy(deep=True)
        else:
            merged_df = merged_df.merge(train_preds_df, how='inner', left_index=True, right_index=True)
        print(merged_df.shape)
    
    merged_train_preds = merged_df.merge(train_df, left_index=True, right_index=True, how='inner')
    
    for model in model_list:
        merged_train_preds[f"{model}_f1_score"] = merged_train_preds.apply(
        lambda x: f1_score(
            x["true_labels"], x[f"{model}_preds"], average="macro"), axis=1,)
        print(f"Average F1 score for {model} train data is \
              {merged_train_preds[f'{model}_f1_score'].mean()*100:0.4f}%")
    
    merged_train_preds.to_parquet('data/merged_train_predictions.parquet')
    
    return merged_train_preds


if __name__ == "__main__":
    # To merge the test prediction files
    
    contesting_models = ['roberta', 'scibert', 'deberta', 'biomed_roberta', 'cs_roberta']
    model_list = contesting_models[:2]
    print(model_list)
    test_pred_df = merge_test_model_predictions(model_list)
