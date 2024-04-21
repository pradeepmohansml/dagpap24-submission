BASELINE="scibert"
BASELINE_CONFIG_FILE_NAME="config_baseline.yml"
DISTILBERT_CONFIG_FILE_NAME="config_distilbert.yml"
ROBERTA_CONFIG_FILE_NAME="config_roberta.yml"
BIOGPT_CONFIG_FILE_NAME="config_biogpt.yml"
T5_CONFIG_FILE_NAME="config_T5.yml"
PHI_CONFIG_FILE_NAME="config_phi.yml"
OPENAI_CONFIG_FILE_NAME="config_openai_gpt2.yml"
FALCON_CONFIG_FILE_NAME="config_falcon.yml"

BASELINE_CONFIG_JSON="config_baseline.json"
DISTILBERT_CONFIG_JSON="config_distilbert.json"
ROBERTA_CONFIG_JSON="config_roberta.json"
BIOGPT_CONFIG_JSON="config_biogpt.json"
T5_CONFIG_JSON="config_T5.json"
PHI_CONFIG_JSON="config_phi.json"
OPENAI_CONFIG_JSON="config_openai_gpt2.json"
FALCON_CONFIG_JSON="config_falcon.json"

CANDIDATE_CONFIG_FILES = {
                          "scibert":BASELINE_CONFIG_FILE_NAME,
                          "distilbert":DISTILBERT_CONFIG_FILE_NAME,
                          "roberta":ROBERTA_CONFIG_FILE_NAME,
                          "biogpt":BIOGPT_CONFIG_FILE_NAME,
                          "t5": T5_CONFIG_FILE_NAME,
                          "phi":PHI_CONFIG_FILE_NAME,
                          "openaigpt2":OPENAI_CONFIG_FILE_NAME,
                          "falcon":FALCON_CONFIG_FILE_NAME
                         }

JUDGES=["GPT35_TURBO","SETFIT_GBC"]
SETFIT_BODY="Salesforce/SFR-Embedding-Mistral"
