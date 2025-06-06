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
Preprocessing script before training DistilBERT.
Specific to BERT -> DistilBERT.
"""
import argparse
import os 
import torch

from transformers import BertForMaskedLM,RobertaForMaskedLM,GPT2LMHeadModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extraction some layers of the full BertForMaskedLM or RObertaForMaskedLM for Transfer Learned"
            " Distillation"
        )
    )
    parser.add_argument("--model_type", default="roberta", choices=["bert","roberta","gpt2"])
    parser.add_argument("--model_name", default="roberta-base", type=str)
    parser.add_argument("--dump_checkpoint", default=None, type=str)
    args = parser.parse_args()

    if args.model_type == "bert":
        model = BertForMaskedLM.from_pretrained(args.model_name)
    elif args.model_type == "roberta":
        model = RobertaForMaskedLM.from_pretrained(args.model_name)
    elif args.model_type == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
    else:
        raise ValueError('args.model_type should be "bert","roberta" or "gpt2".')

    if args.model_type=="roberta":
        if args.model_name=="roberta-base":
        #model.roberta.encoder.layer= model.roberta.encoder.layer[0:12:2]
            model.roberta.encoder.layer = model.roberta.encoder.layer[0:3]+model.roberta.encoder.layer[-3:]
        elif args.model_name=="robera-large":
            model.roberta.encoder.layer = model.roberta.encoder.layer[0:6]+model.roberta.encoder.layer[-6:]
    elif args.model_type=="gpt2":
        if args.model_name=="gpt2-xl":
            model.transformer.h = model.transformer.h[0:2]+model.transformer.h[-2:]
    compressed_sd = model.state_dict()
    if args.model_name == "roberta-base":
        checkpoint_name = f"init_distill_{args.model_name}_first3_last3.pth"
    elif args.model_name == "roberta-large":
        checkpoint_name = f"init_distill_{args.model_name}_first6_last6.pth"
    elif args.model_name == "gpt2-xl":
        checkpoint_name = f"init_distill_{args.model_name}_first2_last2.pth"
    dump_path = os.path.join(args.dump_checkpoint,checkpoint_name)
    print(f"Save transferred checkpoint to {dump_path}.")
    torch.save(compressed_sd, dump_path)