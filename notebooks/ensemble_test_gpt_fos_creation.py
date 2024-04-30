import pandas as pd
import numpy as np
import s3fs
import pyarrow.parquet as pq
import os
import fastparquet
import random
import statistics
import traceback
import ast
from collections import Counter
import langchain
import json
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, List
import re
import string
from openai import RateLimitError, BadRequestError, APITimeoutError
import tiktoken
from dotenv import load_dotenv
import time
import logging
logger = logging.getLogger(__name__)
from tqdm.auto import tqdm
from tqdm import tqdm
from tqdm.gui import tqdm as tqdm_gui
from unittest.mock import patch
load_dotenv("../_envvars.txt")
import math
#Create Batches for GPT FOS creation
def create_batches(input_text_index_dict):
    length_of_data = len(list(input_text_index_dict.keys()))
    batch_size = 100
    nbatches = math.ceil(length_of_data/batch_size)
    batches_of_keys = []
    key_list = list(input_text_index_dict.keys())
    for i in range(0,nbatches):
        a_batch_of_keys = key_list[i*batch_size:(i+1)*batch_size]
        batches_of_keys.append(a_batch_of_keys)
    batches = list(map(lambda x:{key:input_text_index_dict.get(key,"")["text"] for key in x},batches_of_keys))
    return batches

def merged_model_predictions(path=None):
    if 's3://' in path:
        combined_df = pq.ParquetDataset(path, filesystem=s3).read_pandas().to_pandas()
        combined_df["tokens"] = combined_df.tokens.map(lambda x:ast.literal_eval(x.decode()))
    else:
        combined_df = pd.read_parquet(path,engine='fastparquet')
        #combined_df["tokens"] = combined_df.tokens.map(lambda x:ast.literal_eval(x.decode()))   
    return combined_df

#define Pydantic class for Structured output for article field of study
class ArticleFieldOfStudy(BaseModel):
    major_field_of_study: str = Field(description="The major field of study associated with the text of the article")
    sub_areas_within_major_field_of_study: List[str] = Field(description="A list sub areas within the major field of study associated with the text of the article")
    allied_field_of_study: List[str] = Field(description="List of other major fields of study associated with the text of the article")    

#GPT Based Article Field of Study creation    
def populate_field_of_study(batches,load_last_checkpt=None,checkpt=True):
    LLM_MAX_LENGTH = os.getenv('LLM_MAX_LENGTH',15000)
    LLM_MAX_LENGTH=int(LLM_MAX_LENGTH)
    results_batches = []
    rate_limit_delay = 5
    checkpt_dict={}
    i=0
    last_chekpt=None
    while i < len(batches):
        if load_last_checkpt is not None:
            with open(load_last_checkpt,"r+") as f:
                checkpt_dict=json.load(f)
            f.close()
            i = int(load_last_checkpt.strip('.json').split('_')[2])-1
            print(f"Loaded Last checkpointed batch {i+1}")
            results_batches = checkpt_dict[f"Batch_{i+1}"]
            last_chekpt = load_last_checkpt.strip('.json')
            load_last_checkpt = None
            i+=1
            if i<len(batches):
                batch = batches[i]
            else:
                return results_batches, last_chekpt
        else:
            batch = batches[i]
        print(f"Processing Batch {i+1}")
        input_texts=[batch[key] for key in list(batch.keys())]
        try:
            results = chain.batch(input_texts)
        except APITimeoutError:
            checkpt_dict[f"Batch_{i+1}"] = results_batches
            with open(f"checkpt_batch_{i+1}.json","w+") as f:
                json.dump(checkpt_dict,f)
            f.close()
            last_checkpt=f"checkpt_batch_{i+1}"
        except RateLimitError:
            delay = 30
            print(f"Rate Limit Error Encountered, sleeping for {delay} seconds")
            time.sleep(delay)
            results = chain.batch(input_texts)
            rate_limit_delay *=2
            print(f"Doubling rate limit delay between batches to {rate_limit_delay} seconds")
        except BadRequestError:
            print(f"Bad Request Error Hit, adjusting input text length to accomodate model context")
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            encoded_input_texts = list(map(lambda x:encoding.encode(x),input_texts))
            num_tokens_for_texts = list(map(lambda x:len(x),encoded_input_texts))
            clipped_input_texts = []
            for j,encoded_text in enumerate(encoded_input_texts):
                num_tokens = len(encoded_text)
                if num_tokens > LLM_MAX_LENGTH:
                    print(f"In Batch {i+1}, text {j+1} has {num_tokens} tokens, reducing it down to {LLM_MAX_LENGTH}")
                    clipped_input_texts.append(encoding.decode(encoded_text[:LLM_MAX_LENGTH]))
                else:
                    #print(f"In Batch {i+1}, text {j+1} has {num_tokens} tokens, NOT reducing the tokens")
                    clipped_input_texts.append(encoding.decode(encoded_text))
            try:
                results = chain.batch(clipped_input_texts)
            except RateLimitError:
                delay = 30
                print(f"Rate Limit Error Encountered, sleeping for {delay} seconds")
                time.sleep(delay)
                results = chain.batch(clipped_input_texts)
                rate_limit_delay *=2
                print(f"Doubling rate limit delay between batches to {rate_limit_delay} seconds")
        batch_result_dict = {key:results[i] for i,key in enumerate(list(batch.keys()))}
        results_batches.append(batch_result_dict)
        if checkpt:
            checkpt_dict[f"Batch_{i+1}"] = results_batches
            with open(f"checkpt_batch_{i+1}.json","w+") as f:
                json.dump(checkpt_dict,f)
            f.close()
            last_chekpt=f"checkpt_batch_{i+1}"
        if i+2<=len(batches):
            print(f"Sleeping for {rate_limit_delay} seconds before processing batch {i+2}")
        time.sleep(rate_limit_delay)
        if i<len(batches):
            i+=1
    return results_batches, last_chekpt    
print(os.environ['LLM_MAX_LENGTH'])
s3 = s3fs.S3FileSystem()
contesting_models = ['roberta', 'scibert', 'deberta', 'biomed_roberta', 'cs_roberta']
path='data/merged_test_predictions.parquet'
#path = 'data/test_data.parquet'
combined_test_df = merged_model_predictions(path)
print(combined_test_df.columns)
print(combined_test_df.describe())

#Load MAG-FOS Taxonomy JSON for different fields of study"
with open('MAG_FOS.json',"r+") as f:
    mag_fos_taxonomy = json.load(f)
print(mag_fos_taxonomy)
major_fields_of_study = list(map(lambda x:x['field_of_study'],mag_fos_taxonomy["FOS"]))
major_fields_of_study_str = ",".join(major_fields_of_study)
sub_areas_within_major_field_of_study_list = list(map(lambda x:{x['field_of_study']:x['sub_fields']},mag_fos_taxonomy["FOS"]))
sub_areas_within_major_field_of_study = {list(fos.keys())[0]:fos[list(fos.keys())[0]] for fos in sub_areas_within_major_field_of_study_list}
sub_areas_within_major_field_of_study_str = "\n".join(f"{k}:{v}" for k,v in sub_areas_within_major_field_of_study.items())
article_fos_dict_schema = convert_to_openai_tool(ArticleFieldOfStudy)
#Setup and test the LLM Instance for all tasks with respect to this analysis
llm_models = ['gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125']
llms = list(map(lambda x: ChatOpenAI(model=x, temperature=0, max_retries=0,request_timeout=120),llm_models))
llm_tests = list(map(lambda x:x.invoke("who are you, give me your model name and version?"),llms))
print(llms[1].request_timeout)
#Create a ChatPromptTempate for executing 
system = f'''Given an input text from a scientific article identify relevant information about the text.
            You can make use of the following major fields of study: {major_fields_of_study_str}
            You can also make use of the following sub areas within each major field of study listed above: {sub_areas_within_major_field_of_study_str}
         '''
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{input}"),]
)

#construct structured LLMs from input LLMs
structured_llm = llms[1].with_structured_output(article_fos_dict_schema)
structured_article_fos_chain = prompt | structured_llm
print(structured_article_fos_chain)
chain = structured_article_fos_chain
input_text_with_index = pd.DataFrame(combined_test_df['text'])
input_text_with_index_dict = input_text_with_index.to_dict('index')
#create batches for GPT FOS prediction
print("Creating Batches")
batches = create_batches(input_text_with_index_dict)
batch_sizes = [len(b) for b in batches]
print(f"Number of Batches = {len(batches)}")
print(f"Maximum Batch Size = {max(batch_sizes)}, Minimum Batch Size = {min(batch_sizes)}")
print(f"Running Field of Study Population with GPT")
last_checkpt=None
results_batches, last_checkpt = populate_field_of_study(batches,load_last_checkpt=f"{last_checkpt}.json" if last_checkpt is not None else None)
print(last_checkpt)
results_dict = {}
results_dict["results"]=results_batches
with open("gpt_fos_results_test_data.json","w+") as f:
    json.dump(results_dict,f)
f.close()
