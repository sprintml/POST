# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing script before distillation.
"""
import argparse
import logging
import pickle
import random
import time
import os
import numpy as np

from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer
from datasets import load_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument('--datasets',type=str,nargs="+",help="bookcorpus or wikipedia or both")
    parser.add_argument("--tokenizer_type", type=str, default="bert", choices=["bert", "roberta", "gpt2"])
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base", help="The tokenizer to use.")
    parser.add_argument("--dump_path",type=str,help="path to save the dump file")
    
    args = parser.parse_args()

    logger.info(f"Loading Tokenizer ({args.tokenizer_name})")
    if args.tokenizer_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map["cls_token"]  # `[CLS]`
        sep = tokenizer.special_tokens_map["sep_token"]  # `[SEP]`
    elif args.tokenizer_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map["cls_token"]  # `<s>`
        sep = tokenizer.special_tokens_map["sep_token"]  # `</s>`
    elif args.tokenizer_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map["bos_token"]  # `<|endoftext|>`
        sep = tokenizer.special_tokens_map["eos_token"]  # `<|endoftext|>`

    logger.info(f"Loading text from huggingface")
    datasets = args.datasets
    dataset_list=[]
    for set_name in datasets:
        if set_name =="bookcorpus":
            dataset = load_dataset('bookcorpus')['train']
        elif set_name =="wikipedia":
            dataset = load_dataset("wikipedia", "20220301.en")['train']
        dataset_list.append(dataset)

    logger.info("Start encoding")
    num=0
    for dataset in dataset_list:
        num+=len(dataset)
    logger.info(f"{num} examples to process.")

    rslt = []
    iter = 0
    interval = 200000
    start = time.time()
    for dataset in dataset_list:
        for i in range(len(dataset)):
            text = f"{bos} {dataset[i]['text'].strip()} {sep}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            rslt.append(token_ids)

            iter += 1
            if iter % interval == 0:
                end = time.time()
                logger.info(f"{iter} examples processed. - {(end-start):.2f}s/{interval}expl")
                start = time.time()
        logger.info("Finished binarization")
    
    logger.info(f"{num} examples processed.")
    dataset_name = "_".join([d for d in args.datasets])
    dp_name = f"{dataset_name}.{args.tokenizer_name}.pickle"
    dp_file = os.path.join(args.dump_path,dp_name)
    vocab_size = tokenizer.vocab_size
    if vocab_size < (1 << 16):
        rslt_ = [np.uint16(d) for d in rslt]
    else:
        rslt_ = [np.int32(d) for d in rslt]
    random.shuffle(rslt_)
    logger.info(f"Dump to {dp_file}")
    with open(dp_file, "wb") as handle:
        pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()