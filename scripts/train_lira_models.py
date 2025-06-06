import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import argparse
import torch.nn.functional as F
import torch
import torch.nn as nn
from dataloaders.sst_loader import SST2_Dataset,get_rmia_sst2_loader
from dataloaders.imdb_loader import IMDB_Dataset,get_rmia_imdb_loader
from dataloaders.arisetv_loader import Arisetv_Dataset,get_rmia_arisetv_loader
from dataloaders.tweet_eval_loader import TweetEval_sentiment_Dataset,get_rmia_tweeteval_sentiment_loader
from transformers import RobertaForMaskedLM, RobertaTokenizer,RobertaConfig,AutoModelForCausalLM,AutoTokenizer,GPT2Config,LlamaConfig,AutoModelForCausalLM
from scripts.constants import TEXT_LABEL_DICT,INIT_TEXT_DICT
#from utils.database_utils import get_config, insert_result, connect
from peft import get_peft_model, PromptEmbedding, PromptTuningConfig, TaskType
from model.torch_models import Lit_RobertaMLM_text_infilling
from pytorch_lightning import Trainer
from opacus.data_loader import DPDataLoader
import numpy as np


os.environ["TOKENIZERS_PARALLELISM"] = "false"
def none_or_str(value):
    if value == 'None':
        return None
    return value

def get_args():
    parser = argparse.ArgumentParser()
    # model and datasets
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base')
    parser.add_argument('--config_path',type=none_or_str,default=None)
    parser.add_argument('--model_state',type=none_or_str,default=None)
    parser.add_argument('--dataset',type=str,default="sst2")
    # hyperparameters
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--max_length',type=int,default=128)
    parser.add_argument('--num_tokens',type=int,default=100)
    parser.add_argument('--n_epochs',type=int,default=5)
    parser.add_argument('--optimizer',type=str,default="Adam")
    parser.add_argument('--lr',type=float,default=5e-4)
    parser.add_argument('--seed',type=int,default=42)
    
    parser.add_argument('--db_config_path',type=none_or_str,default=None)
    parser.add_argument('--collection_name',type=str,default=None)

    # result save
    parser.add_argument('--output_dir',type=str,default=None)
    parser.add_argument('--token_path',type=str,default=None,help="path to huggingface token")
    parser.add_argument('--temp',type=float,default=1.5,help="temperature for synthetic data, only for imdb_syn")
    parser.add_argument('--weight_decay',type=float,default=0.0,help="weight decay for optimizer")
    parser.add_argument('--epsilon',type=float,default=8.0,help="epsilon for DP")
    parser.add_argument('--shadow_N',type=int,default=8,help="shadow id for RMIA")
    parser.add_argument('--shadow_id',type=int,default=0,help="shadow id for RMIA")
    args = parser.parse_args()
    return args


def main(args):
    # step1 get model and tokenizer
    print(args)
    seed= args["seed"]
    torch.manual_seed(seed)
    import huggingface_hub
    def load_token(path):
        with open(path,"r") as f:
            token = f.read().strip()
        return token
    token = load_token(args["token_path"])
    huggingface_hub.login(token=token)
    model_name = args["model_name"]
    if "gpt" in model_name or "Llama" in model_name:
        task="clm"
    else:
        task="mlm"
    tokenizer_name = args["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if args["config_path"]: # load from config and pretrained weights
        config_file= args["config_path"]
        state_file = args["model_state"]
        if "gpt" in model_name:
            model_config = GPT2Config.from_json_file(config_file)
            model = AutoModelForCausalLM.from_config(model_config)
        elif "Llama" in model_name:
            model_config = LlamaConfig.from_json_file(config_file)
            model = AutoModelForCausalLM.from_config(model_config)
        elif "roberta" in model_name:
            model_config = RobertaConfig.from_json_file(config_file)
            model = RobertaForMaskedLM(model_config)
        model.load_state_dict(torch.load(state_file))
    else:
        if "gpt" in model_name or "Llama" in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name,use_auth_token=True)
        else:
            model = RobertaForMaskedLM.from_pretrained(model_name)

    # step2 get dataset
    dataset_name = args["dataset"]
    batch_size = args['batch_size']
    max_length = args["max_length"]
    #assert dataset_name == "sst2"
    shadow_id = args['shadow_id']
    shadow_N = args['shadow_N']
    epsilon = args["epsilon"]
    if dataset_name == "sst2":
        train_loader,keep_bool = get_rmia_sst2_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=True,max_length=max_length,task=task,shadow_N=shadow_N,shadow_id=shadow_id)
    elif dataset_name == "imdb":
        train_loader,keep_bool = get_rmia_imdb_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=True,max_length=max_length,task=task,shadow_N=shadow_N,shadow_id=shadow_id)
    elif dataset_name == "arisetv":
        train_loader,keep_bool = get_rmia_arisetv_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=True,max_length=max_length,task=task,shadow_N=shadow_N,shadow_id=shadow_id)
    elif dataset_name == "tweeteval_sentiment":
        train_loader,keep_bool = get_rmia_tweeteval_sentiment_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=True,max_length=max_length,task=task,shadow_N=shadow_N,shadow_id=shadow_id)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    #print(next(iter(train_loader)))
    #if epsilon<1000: # DP
    #    train_loader = DPDataLoader.from_data_loader(train_loader)
    
    # step3 wrap model to peft model
    model_config_dict = model.config.to_dict()
    if "gpt" in model_name:
        token_dim = model_config_dict['n_embd']
        num_attention_heads= model_config_dict['n_head']
        num_layers = model_config_dict['n_layer']
    elif "Llama" in model_name:
        token_dim = model_config_dict['hidden_size']
        num_attention_heads = model_config_dict['num_attention_heads']
        num_layers = model_config_dict['num_hidden_layers']
    else:
        token_dim = model_config_dict['hidden_size']
        num_attention_heads = model_config_dict['num_attention_heads']
        num_layers = model_config_dict['num_hidden_layers']
        
    num_tokens = args["num_tokens"]
    
    pt_config = PromptTuningConfig(
        peft_type="PROMPT_TUNING",
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_tokens,
        token_dim=token_dim,
        num_transformer_submodules=1,
        num_attention_heads=num_attention_heads,
        num_layers=num_layers,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text=INIT_TEXT_DICT[dataset_name],
        tokenizer_name_or_path=tokenizer_name,
        )
    peft_model = get_peft_model(model,pt_config)
    peft_model.print_trainable_parameters()
    
    # step4 wrap peft model to lit_model
    lr = args["lr"]
    optimizer = args['optimizer']
    n_epochs = args["n_epochs"]
    weight_decay = args["weight_decay"]

    delta = 1./(len(train_loader)*batch_size)

    output_dir = args["output_dir"]
    if "Llama" in model_name:
        model_name = "Llama-2-7b-hf"
    file_name = f"{model_name}_{dataset_name}_{seed}_{lr}_{n_epochs}_{optimizer}_{epsilon}_{shadow_N}_{shadow_id}"
    output_dir = os.path.join(output_dir,file_name)
    os.makedirs(output_dir,exist_ok=True)
    output_name = f"prompt_state.pth"
    prompt_dir = os.path.join(output_dir,output_name)
    if not os.path.exists(prompt_dir):
        print(f"Prompt directory {prompt_dir} doesn't exist.")
        # train model and save prompt
        if epsilon<1000:
            print("Enable DP")
            lit_model = Lit_RobertaMLM_text_infilling(peft_model,
                                                    tokenizer=tokenizer,
                                                    lr =lr,
                                                    optimizer=optimizer,
                                                    dataset = dataset_name,
                                                    text_label_dict=TEXT_LABEL_DICT,
                                                    token_num=num_tokens,
                                                    weight_decay=weight_decay,
                                                    enable_dp=True,
                                                    delta = delta,
                                                    epsilon = epsilon
                                                    )
        else:
            lit_model = Lit_RobertaMLM_text_infilling(peft_model,
                                                    tokenizer=tokenizer,
                                                        lr =lr,
                                                        optimizer=optimizer,
                                                        dataset = dataset_name,
                                                        text_label_dict=TEXT_LABEL_DICT,
                                                        token_num=num_tokens,
                                                        weight_decay=weight_decay,
                                                        enable_dp=False,
                                                    )
        trainer = Trainer(max_epochs=n_epochs,accelerator="gpu",enable_progress_bar=True,fast_dev_run=False)

        trainer.fit(lit_model,train_loader)
        trainer.test(lit_model,train_loader)
        prompt = lit_model.model.prompt_encoder
        torch.save(prompt.state_dict(),prompt_dir)
        peft_model = lit_model.model
        
        idx_output_dir = f"{output_dir}/keep.npy"
        np.save(idx_output_dir, keep_bool)
    else:
        # load state_dict
        print(f"Prompt directory {prompt_dir} exists.")
        loaded_prompt = torch.load(prompt_dir)
        state = peft_model.state_dict()
        state['prompt_encoder.default.embedding.weight'] = loaded_prompt['default.embedding.weight']
        peft_model.load_state_dict(state)
        idx_output_dir = f"{output_dir}/keep.npy"
        np.save(idx_output_dir, keep_bool)
    peft_model.to("cuda")

    
    # inference
    # get normal SST2 dataset
    if dataset_name == "sst2":
        train_dataset = SST2_Dataset(tokenizer,"train",text_infilling=True,max_length=max_length,seed=None,task=task)
    elif dataset_name == "imdb":
        train_dataset = IMDB_Dataset(tokenizer,"train",text_infilling=True,max_length=max_length,seed=None,task=task)
    elif dataset_name == "arisetv":
        train_dataset = Arisetv_Dataset(tokenizer,"train",text_infilling=True,max_length=max_length,seed=None,task=task)
    elif dataset_name == "tweeteval_sentiment":
        train_dataset = TweetEval_sentiment_Dataset(tokenizer,"train",text_infilling=True,max_length=max_length,seed=None,task=task)        
    
    data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
    peft_model.eval()
    saved_logits = []
    data_label_idx = []
    text_labels =TEXT_LABEL_DICT[dataset_name]
    for text_label_list in text_labels:
        idx_list=[tokenizer.encode(text_label)[1] for text_label in text_label_list]
        data_label_idx.append(idx_list)
    for batch in data_loader:
        logits = peft_model(batch["input_ids"].to("cuda"),batch['attention_mask'].to("cuda")).logits.detach().cpu()
        mask_idx = batch['mask_idx']+num_tokens
        label = batch['label']
    
        for logit,idx,lab in zip(logits,mask_idx,label):
            target_logit = logit[idx]
            pred_logits = torch.hstack([target_logit[index].mean() for index in data_label_idx])
            #print(pred_logits)
            pred_logits = F.softmax(pred_logits, dim=-1)
            #print(pred_logits)
            saved_logits.append(pred_logits[lab])
    save_logits = np.array(saved_logits)
    loss_name = "inference_loss.npy"
    loss_path = os.path.join(output_dir,loss_name)
    np.save(loss_path,save_logits)


if __name__=="__main__":
    args = get_args()
    args = vars(args)
    main(args)