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
from dataloaders.sst_loader import get_sst2_loader,SST2_Dataset
from dataloaders.imdb_loader import get_imdb_loader,IMDB_Dataset
from dataloaders.tweet_eval_loader import get_tweeteval_sentiment_loader,TweetEval_sentiment_Dataset
from dataloaders.fpb_loader import get_fpb_loader,FPB_Dataset,get_data_and_split
from dataloaders.mnli_loader import get_mnli_loader,MNLI_Dataset
from dataloaders.qnli_loader import get_qnli_loader,QNLI_Dataset
from dataloaders.snli_loader import get_snli_loader,SNLI_Dataset
from dataloaders.agnews_loader import get_agnews_loader,AgNews_Dataset
from dataloaders.arisetv_loader import get_arisetv_loader,Arisetv_Dataset
from dataloaders.dbpedia_loader import get_dbpedia_loader,DBPedia_Dataset
from dataloaders.trec_loader import get_trec_loader,Trec_Dataset
from transformers import RobertaForMaskedLM,RobertaConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config
from scripts.constants import TEXT_LABEL_DICT,INIT_TEXT_DICT

from peft import get_peft_model, PromptTuningConfig, TaskType
from peft import AutoPeftModelForCausalLM
from model.torch_models import Lit_RobertaMLM_text_infilling
from pytorch_lightning import Trainer
from utils.transfer import absolute_transfer,relative_transfer,translation_transfer,evaluate,translation_after_prompt_transfer,absolute_shift_transfer,topk_transfer,topk_output,inference_transfer,mix_transfer


def none_or_str(value):
    if value == 'None':
        return None
    return value

def get_args():
    parser = argparse.ArgumentParser()
    # model and datasets
    parser.add_argument('--target_model_name', type=str, default='roberta-base')
    parser.add_argument('--target_model_path', type=none_or_str, default=None)
    parser.add_argument('--tokenizer_name', type=str, default='roberta-base')
    parser.add_argument('--tokenizer_path',type=none_or_str,default=None)
    parser.add_argument('--source_model_name',type=str,default='roberta-base')
    parser.add_argument('--source_path',type=str,default=None,help="source prompt path")
    parser.add_argument('--source_model_config_path',type=none_or_str,default=None)
    parser.add_argument('--source_model_state',type=none_or_str,default=None)
    parser.add_argument('--target_dataset',type=str,default=None)
    parser.add_argument('--target_dataset_path',type=none_or_str,default=None)
    parser.add_argument('--anchor_dataset',type=str,default=None)
    parser.add_argument('--anchor_dataset_path',type=none_or_str,default=None)
    parser.add_argument('--anchor_num',type=int,default=1000)

    # hyperparameters
    parser.add_argument('--num_tokens',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--n_steps',type=int,default=6000)
    parser.add_argument('--accumulate_step',type=int,default=1)
    parser.add_argument('--init_with_source',action="store_true")
    parser.add_argument('--interval',type=int,default=500)
    parser.add_argument('--optimizer',type=str,default="Adam")
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--method',type=str,default="output")
    parser.add_argument('--feature_token',type=str,default="mask")
    parser.add_argument('--loss_name',type=str,default="KLDiv")
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max_length',type=int,default=128)
    parser.add_argument('--topk',type=int,default=10)
    parser.add_argument("--alpha",type=float,default=0.5)
    # database related
    parser.add_argument('--db_config_path',type=none_or_str,default=None)
    parser.add_argument('--collection_name',type=str,default=None)
    parser.add_argument('--lr_scheduler',action="store_true")
    parser.add_argument('--batch_num',type=int,default=0)

    # result save
    parser.add_argument('--output_dir',type=str,default=None)
    args = parser.parse_args()
    return args

def get_dataloaders(dataset_name,tokenizer,batch_size,text_infilling=True,local_dir=None,max_length=None,seed=None,task="mlm"):
    if dataset_name=="sst2":
        train_loader,val_loader = get_sst2_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,seed=seed,task=task)
    elif dataset_name=="imdb"
        train_loader,val_loader = get_imdb_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,seed=seed,task=task)
    elif dataset_name =="tweeteval_sentiment":
        train_loader,val_loader = get_tweeteval_sentiment_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,seed=seed,task=task)
    elif dataset_name =="qnli":
        train_loader,val_loader = get_qnli_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length)
    elif dataset_name =="snli":
        train_loader,val_loader = get_snli_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length)
    elif dataset_name =="agnews":
        train_loader,val_loader = get_agnews_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,seed=seed,task=task)
    elif dataset_name=="arisetv":
        train_loader,val_loader = get_arisetv_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,seed=seed,task=task)
    elif dataset_name=="dbpedia":
        train_loader,val_loader = get_dbpedia_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length,task=task)
    elif dataset_name=="trec":
        train_loader,val_loader = get_trec_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length,task=task)
    return train_loader,val_loader

def get_dataset(dataset_name, tokenizer,text_infilling=False,local_dir=None,max_length=None,seed=None,task="mlm"):
    if dataset_name=="sst2":
        dataset = SST2_Dataset(tokenizer=tokenizer,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,split="train",seed=seed,task=task)
    elif dataset_name=="imdb":
        dataset = IMDB_Dataset(tokenizer=tokenizer,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,split="train",seed=seed,task=task)
    elif dataset_name=="tweeteval_sentiment":
        dataset = TweetEval_sentiment_Dataset(tokenizer=tokenizer,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,split="train",seed=seed,task=task)
    elif dataset_name=="qnli":
        dataset = QNLI_Dataset(tokenizer=tokenizer,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,split="train",seed=seed)
    elif dataset_name=="snli":
        dataset = SNLI_Dataset(tokenizer=tokenizer,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,split="train",seed=seed)
    elif dataset_name =="agnews":
        dataset = AgNews_Dataset(tokenizer=tokenizer,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,split="train",seed=seed,task=task)
    elif dataset_name=="arisetv":
        dataset = Arisetv_Dataset(tokenizer=tokenizer,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,split="train",seed=seed,task=task)
    elif dataset_name=="dbpedia":
        dataset = DBPedia_Dataset(ttokenizer=tokenizer,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,split="train",seed=seed,task=task)
    elif dataset_name=="trec":
        dataset = Trec_Dataset(tokenizer=tokenizer,text_infilling=text_infilling,local_dir=local_dir,max_length=max_length,split="train",seed=seed,task=task)
    return dataset

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device="cuda"
    seed= args["seed"]
    torch.manual_seed(seed)

    # step1: get target model and tokenizer
    target_model_name = args["target_model_name"]
    target_model_path = args["target_model_path"]
    if "gpt" in target_model_name:
        if target_model_path:
            target_model = AutoModelForCausalLM.from_pretrained(target_model_path)
        else:
            target_model = AutoModelForCausalLM.from_pretrained(target_model_name)
    else:
        if target_model_path:
            target_model = RobertaForMaskedLM.from_pretrained(target_model_path).to(device)
        else:
            target_model = RobertaForMaskedLM.from_pretrained(target_model_name).to(device)
    tokenizer_name = args["tokenizer_name"]
    tokenizer_path = args["tokenizer_path"]
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # step2: wrap target model to prompt tuning model
    model_config_dict = target_model.config.to_dict()
    if "gpt" in target_model_name:
        token_dim = model_config_dict['n_embd']
        num_attention_heads= model_config_dict['n_head']
        num_layers = model_config_dict['n_layer']
        task="clm"
    else:
        token_dim = model_config_dict['hidden_size']
        num_attention_heads = model_config_dict['num_attention_heads']
        num_layers = model_config_dict['num_hidden_layers']
        task="mlm"
        
    num_tokens = args["num_tokens"]
    target_dataset = args["target_dataset"]
    pt_config = PromptTuningConfig(
        peft_type="PROMPT_TUNING",
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_tokens,
        token_dim=token_dim,
        num_transformer_submodules=1,
        num_attention_heads=num_attention_heads,
        num_layers=num_layers,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text=INIT_TEXT_DICT[target_dataset],
        tokenizer_name_or_path=tokenizer_name,
        )
    target_peft_model = get_peft_model(target_model,pt_config).to(device)
    target_peft_model.print_trainable_parameters()
    
    # step3: get target dataset, only need dev set
    batch_size = args["batch_size"]
    max_length = args['max_length']
    target_dataset_path = args["target_dataset_path"]
    _,target_loader = get_dataloaders(dataset_name=target_dataset,
                                      tokenizer=tokenizer,
                                      batch_size = batch_size,
                                      local_dir=target_dataset_path,
                                      text_infilling=True,
                                      max_length=max_length,
                                      seed=seed,
                                      task=task)
    
    # step3: get source prompt and source model if needed
    source_model_name = args["source_model_name"]
    source_model_path = args["source_path"]
    
    if args["source_model_config_path"]:
        config_file= args["source_model_config_path"]
        state_file = args["source_model_state"]
        if "gpt" in source_model_name:
            model_config = GPT2Config.from_json_file(config_file)
            source_model = AutoModelForCausalLM.from_config(model_config)
        else:
            model_config = RobertaConfig.from_json_file(config_file)
            source_model = RobertaForMaskedLM(model_config).to(device)
        source_model.load_state_dict(torch.load(state_file))
        source_peft_model = get_peft_model(source_model,pt_config)
        loaded_prompt_encoder_state = torch.load(source_model_path)
        # load state dict 
        source_state = source_peft_model.state_dict()
        source_state['prompt_encoder.default.embedding.weight'] = loaded_prompt_encoder_state['default.embedding.weight']
        source_peft_model.load_state_dict(source_state)
        source_peft_model = source_peft_model.to(device)
    else:
        if "gpt" in source_model_name:
            source_model = AutoModelForCausalLM.from_pretrained(source_model_name).to(device)
            source_peft_model = AutoPeftModelForCausalLM.from_pretrained(source_model_path).to(device)
        else:
            source_model = RobertaForMaskedLM.from_pretrained(source_model_name).to(device)
            source_peft_model = AutoPeftModelForCausalLM.from_pretrained(source_model_path).to(device)

    # step4: get anchor dataset if needed
    anchor_dataset = args["anchor_dataset"]
    anchor_dataset_path = args["anchor_dataset_path"]
    anchor_loader,_ = get_dataloaders(dataset_name=anchor_dataset,
                                      tokenizer=tokenizer,
                                      batch_size = batch_size,
                                      text_infilling=True,
                                      local_dir=anchor_dataset_path,
                                      max_length=max_length,
                                      seed=seed,
                                      task=task)
    anchor_set = get_dataset(anchor_dataset, tokenizer,text_infilling=True,local_dir=anchor_dataset_path,max_length=128,seed=args["seed"],task=task)

    # step6: prepare anchors if we use relative_space methods
    method = args["method"]
    optimizer = args["optimizer"]
    # get anchore if relative space methods   
    if "relative" in method:
        anchor_num = args["anchor_num"]
        feature_token = args["feature_token"]
        anchors_list = []
        for i in range(anchor_num):
            anchors_list.append(anchor_set[-1-i])
        source_anchors = []
        target_anchors = []
        with torch.no_grad():
            for anchor in anchors_list:
                x = {"input_ids":anchor['input_ids'].unsqueeze(0).to(device),"attention_mask":anchor['attention_mask'].unsqueeze(0).to(device)}
                idx = anchor['mask_idx']
                if feature_token =="cls":
                    source_anchors.append(source_model(**x,output_hidden_states=True).hidden_states[-1][0,0,:]) # take the hidden state of <cls> token
                    target_anchors.append(target_model(**x,output_hidden_states=True).hidden_states[-1][0,0,:])     
                elif feature_token =="mask":
                    source_anchors.append(source_model(**x,output_hidden_states=True).hidden_states[-1][0,idx,:]) # take the hidden state of <mask> token,
                    target_anchors.append(target_model(**x,output_hidden_states=True).hidden_states[-1][0,idx,:]) 
        source_anchors = torch.vstack(source_anchors)
        target_anchors = torch.vstack(target_anchors)
        # normalize anchors
        source_anchors = F.normalize(source_anchors, p=2, dim=-1)
        target_anchors = F.normalize(target_anchors, p=2, dim=-1)

    # step7: transfer
    n_steps = args["n_steps"] # tranfer traning steps
    lr = args["lr"]
    interval = args["interval"]
    feature_token = args["feature_token"]
    loss_name = args["loss_name"]
    accumulate_step  = args["accumulate_step"]
    init_with_source = args["init_with_source"]
    lr_scheduler=args["lr_scheduler"]
    topk = args["topk"]

    if init_with_source:
        target_peft_model.prompt_encoder=source_peft_model.prompt_encoder

    if method=="direct": # direct transfer, needs to make sure the embedding dim matched
        best_prompt = source_peft_model.prompt_encoder
        target_peft_model.prompt_encoder=source_peft_model.prompt_encoder
        best_acc = evaluate(target_peft_model,target_dataset,tokenizer,target_loader,n_soft=num_tokens,task=task)
        last_acc = best_acc
        best_step = 0
        acc_dict=None
        loss_dict=None

    elif method in ["embedding","output"]: # transfer on the embedding space, needs to make sure the embedding dim mached 
        prefix = f"{batch_size}_{lr}"
        best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict = absolute_transfer(source_peft_model=source_peft_model,target_peft_model=target_peft_model,tokenizer=tokenizer,
                                                                 dataloader=anchor_loader,target_loader=target_loader,target_dataset_name=target_dataset,anchor_dataset_name=anchor_dataset,
                                                                 n_steps=n_steps,method=method,feature_token=feature_token,num_tokens=num_tokens,lr=lr,interval=interval,loss_name=loss_name,prefix=prefix,task=task,
                                                                 accumulate_step=accumulate_step,lr_scheduler=lr_scheduler,optimizer=optimizer)
    elif method in ["embedding_shift","output_shift"]:
        prefix = f"{batch_size}_{lr}_abs_shift"
        best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict = absolute_shift_transfer(source_model=source_model,target_model=target_model,source_peft_model=source_peft_model,target_peft_model=target_peft_model,tokenizer=tokenizer,
                                                                 dataloader=anchor_loader,target_loader=target_loader,target_dataset_name=target_dataset,anchor_dataset_name=anchor_dataset,
                                                                 n_steps=n_steps,method=method,feature_token=feature_token,num_tokens=num_tokens,lr=lr,interval=interval,loss_name=loss_name,prefix=prefix,task=task,
                                                                 accumulate_step=accumulate_step,lr_scheduler=lr_scheduler,optimizer=optimizer)
    elif method in ["topk_output"]:
        prefix = f"{batch_size}_{lr}"
        best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict = topk_output(source_peft_model=source_peft_model,target_peft_model=target_peft_model,tokenizer=tokenizer,
                                                                 dataloader=anchor_loader,target_loader=target_loader,target_dataset_name=target_dataset,anchor_dataset_name=anchor_dataset,
                                                                 n_steps=n_steps,method=method,feature_token=feature_token,num_tokens=num_tokens,lr=lr,interval=interval,loss_name=loss_name,prefix=prefix,task=task,
                                                                 accumulate_step=accumulate_step,lr_scheduler=lr_scheduler,optimizer=optimizer,topk=topk)
      
    elif method in ["topk_transfer"]:
        prefix = f"{batch_size}_{lr}"
        best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict = topk_transfer(source_model=source_model,target_model=target_model,source_peft_model=source_peft_model,target_peft_model=target_peft_model,tokenizer=tokenizer,
                                                                 dataloader=anchor_loader,target_loader=target_loader,target_dataset_name=target_dataset,anchor_dataset_name=anchor_dataset,
                                                                 n_steps=n_steps,method=method,feature_token=feature_token,num_tokens=num_tokens,lr=lr,interval=interval,loss_name=loss_name,prefix=prefix,task=task,
                                                                 accumulate_step=accumulate_step,lr_scheduler=lr_scheduler,optimizer=optimizer,topk=topk)
    
    elif method in ["relative","relative_shift"]: # transfer in relative space, need to specify anchor datasets
        best_prompt,best_acc,best_step,last_acc = relative_transfer(source_model=source_model,target_model=target_model,
                                              source_peft_model=source_peft_model,target_peft_model=target_peft_model,tokenizer=tokenizer,
                                              dataloader=anchor_loader,target_loader=target_loader,target_dataset_name=target_dataset,
                                              source_anchors=source_anchors,target_anchors=target_anchors,
                                              n_steps=n_steps,method=method,feature_token=feature_token,num_tokens=num_tokens,lr=lr,interval=interval)

    elif method=="translation": # tranfer baseed on a translation model of two embedding space, doesn't need to be same dim
        best_prompt,best_acc,best_step,last_acc = translation_transfer(source_model=source_model,target_model=target_model,
                                              source_peft_model=source_peft_model,target_peft_model=target_peft_model,tokenizer=tokenizer,
                                              dataloader=anchor_loader,target_loader=target_loader,target_dataset_name=target_dataset,
                                              n_steps=n_steps,feature_token=feature_token,num_tokens=num_tokens,lr=lr,interval=interval)
    elif method =="translation_after_prompt":
        best_prompt,best_acc,best_step,last_acc = translation_after_prompt_transfer(source_peft_model=source_peft_model,target_peft_model=target_peft_model,tokenizer=tokenizer,
                                                                                    dataloader=anchor_loader,target_loader=target_loader,target_dataset_name=target_dataset,
                                                                                    n_steps=n_steps,feature_token=feature_token,num_tokens=num_tokens,lr=lr,interval=interval)
    elif method =="inference_transfer":
        best_acc = inference_transfer(source_model=source_model,target_model=target_model,source_peft_model=source_peft_model,tokenizer=tokenizer,
                                 target_loader = target_loader,target_dataset_name=target_dataset,num_tokens=num_tokens,task=task)
        best_step=0
        last_acc = best_acc
        acc_dict=None
        loss_dict=None
        best_prompt=None

    elif method == "mix_transfer":
        alpha = args['alpha']
        batch_num = args['batch_num']
        prefix = f"{batch_size}_{lr}_abs_shift"
        best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict = mix_transfer(source_model=source_model,target_model=target_model,source_peft_model=source_peft_model,target_peft_model=target_peft_model,tokenizer=tokenizer,
                                                                 dataloader=anchor_loader,target_loader=target_loader,target_dataset_name=target_dataset,anchor_dataset_name=anchor_dataset,
                                                                 n_steps=n_steps,method=method,feature_token=feature_token,num_tokens=num_tokens,lr=lr,interval=interval,loss_name=loss_name,prefix=prefix,task=task,
                                                                 accumulate_step=accumulate_step,lr_scheduler=lr_scheduler,optimizer=optimizer,alpha=alpha,batch_num=batch_num)
    # step8: double check the final performance
    target_peft_model.prompt_encoder = best_prompt
    
    # step9: upload result to database
    res = dict(args)
    res["best_acc"]=best_acc
    res["best_step"]=best_step
    res["last_acc"]=last_acc
    res["acc_dict"] = acc_dict
    res["loss_dict"] = loss_dict
    
    print(res)

if __name__=="__main__":
    args = get_args()
    args = vars(args)
    print(args)
    main(args)