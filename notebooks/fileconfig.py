BASELINE="scibert"
DISTILBERT_CONFIG_FILE_NAME="config_distilbert.yml"
ROBERTA_CONFIG_FILE_NAME="config_roberta.yml"
BIOMED_ROBERTA_CONFIG_FILE_NAME="config_biomed_roberta.yml"
T5_CONFIG_FILE_NAME="config_T5.yml"
OPENAI_CONFIG_FILE_NAME="config_openai_gpt2.yml"
DEBERTA_CONFIG_FILE_NAME="config_deberta.yml"
SCIBERT_CONFIG_FILE_NAME = "config_scibert.yml"

SCIBERT_CONFIG_JSON = "config_scibert.json"
DISTILBERT_CONFIG_JSON="config_distilbert.json"
ROBERTA_CONFIG_JSON="config_roberta.json"
BIOMED_ROBERTA_CONFIG_JSON="config_biomed_roberta.json"
T5_CONFIG_JSON="config_T5.json"
DEBERTA_CONFIG_JSON="config_deberta.json"
OPENAI_CONFIG_JSON="config_openai_gpt2.json"

CANDIDATE_CONFIG_FILES = {
                          "scibert":SCIBERT_CONFIG_FILE_NAME,
                          "distilbert":DISTILBERT_CONFIG_FILE_NAME,
                          "roberta":ROBERTA_CONFIG_FILE_NAME,
                           "biomed_roberta": BIOMED_ROBERTA_CONFIG_FILE_NAME,
                           "deberta": DEBERTA_CONFIG_FILE_NAME,
                          "t5": T5_CONFIG_FILE_NAME,
                          "openaigpt2":OPENAI_CONFIG_FILE_NAME,
                         }

JUDGES=["GPT35_TURBO","SETFIT_GBC"]
SETFIT_BODY="Salesforce/SFR-Embedding-Mistral"
