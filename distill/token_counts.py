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
Preprocessing script before training the distilled model.
"""
import os
import argparse
import logging
import pickle
from collections import Counter


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Token Counts for smoothing the masking probabilities in MLM (cf XLM/word2vec)"
    )
    parser.add_argument( "--datafile", type=str, help="The binarized dataset.")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base", help="The tokenizer to use.")
    parser.add_argument("--token_counts_dump_path", type=str, help="The dump file.")
    parser.add_argument('--datasets',type=str,nargs="+",help="bookcorpus or wikipedia or both")

    args = parser.parse_args()

    logger.info(f"Loading data from {args.datafile}")
    with open(args.datafile, "rb") as fp:
        data = pickle.load(fp)

    logger.info("Counting occurrences for MLM.")
    counter = Counter()
    for tk_ids in data:
        counter.update(tk_ids)
    if "roberta" in args.tokenizer_name:
        vocab_size = 50265
    elif "bert" in args.tokenizer_name:
        vocab_size=30522
    elif "gpt2" in args.tokenizer_name:
        vocab_size = 50257

    counts = [0] * vocab_size
    for k, v in counter.items():
        counts[k] = v
    dataset_name = "_".join([d for d in args.datasets])
    dump_name = f"token_counts_{dataset_name}_{args.tokenizer_name}.pickle"
    dump_path = os.path.join(args.token_counts_dump_path,dump_name)
    logger.info(f"Dump to {dump_path}")
    with open(dump_path, "wb") as handle:
        pickle.dump(counts, handle, protocol=pickle.HIGHEST_PROTOCOL)