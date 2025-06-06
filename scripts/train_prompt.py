import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import torch.nn.functional as F
import torch
import torch.nn as nn
from dataloaders.sst_loader import get_sst2_loader
from dataloaders.imdb_loader import get_imdb_loader
from dataloaders.tweet_eval_loader import get_tweeteval_sentiment_loader
from dataloaders.fpb_loader import get_fpb_loader
from dataloaders.mnli_loader import get_mnli_loader
from dataloaders.qnli_loader import get_qnli_loader
from dataloaders.snli_loader import get_snli_loader
from dataloaders.agnews_loader import get_agnews_loader
from dataloaders.arisetv_loader import get_arisetv_loader
from dataloaders.dbpedia_loader import get_dbpedia_loader
from dataloaders.trec_loader import get_trec_loader
from transformers import RobertaForMaskedLM, RobertaTokenizer,RobertaConfig,AutoModelForCausalLM,AutoTokenizer,GPT2Config
from scripts.constants import TEXT_LABEL_DICT,INIT_TEXT_DICT
from utils.database_utils import get_config, insert_result, connect
from peft import get_peft_model, PromptEmbedding, PromptTuningConfig, TaskType
from model.torch_models import Lit_RobertaMLM_text_infilling
from pytorch_lightning import Trainer

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
    parser.add_argument('--task',type=str,default="mlm")
    
    parser.add_argument('--db_config_path',type=none_or_str,default=None)
    parser.add_argument('--collection_name',type=str,default=None)

    # result save
    parser.add_argument('--output_dir',type=str,default=None)
    args = parser.parse_args()
    return args


def get_dataloaders(dataset_name,tokenizer,batch_size,text_infilling=True,max_length=None,task="mlm"):
    if dataset_name=="sst2":
        train_loader,val_loader = get_sst2_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length,task=task)
    elif dataset_name=="imdb":
        train_loader,val_loader = get_imdb_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length,task=task)
    elif dataset_name =="tweeteval_sentiment":
        train_loader,val_loader = get_tweeteval_sentiment_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length,task=task)
    elif dataset_name =="qnli":
        train_loader,val_loader = get_qnli_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length)
    elif dataset_name =="snli":
        train_loader,val_loader = get_snli_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length)
    elif dataset_name =="agnews":
        train_loader,val_loader = get_agnews_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length,task=task)
    elif dataset_name=="arisetv":
        train_loader,val_loader = get_arisetv_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length,task=task)
    elif dataset_name=="dbpedia":
        train_loader,val_loader = get_dbpedia_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length,task=task)
    elif dataset_name=="trec":
        train_loader,val_loader = get_trec_loader(tokenizer=tokenizer,batch_size=batch_size,text_infilling=text_infilling,max_length=max_length,task=task)
    return train_loader,val_loader

def main(args):
    # step1 get model and tokenizer
    seed= args["seed"]
    torch.manual_seed(seed)

    model_name = args["model_name"]
    tokenizer_name = args["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if args["config_path"]: # load from config and pretrained weights
        config_file= args["config_path"]
        state_file = args["model_state"]
        if "gpt" in model_name:
            model_config = GPT2Config.from_json_file(config_file)
            model = AutoModelForCausalLM.from_config(model_config)
        else:
            model_config = RobertaConfig.from_json_file(config_file)
            model = RobertaForMaskedLM(model_config)
        model.load_state_dict(torch.load(state_file))
    else:
        if "gpt" in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = RobertaForMaskedLM.from_pretrained(model_name)

    # step2 get dataset
    dataset_name = args["dataset"]
    batch_size = args['batch_size']
    max_length = args["max_length"]
    task = args["task"]
    train_loader,val_loader = get_dataloaders(dataset_name=dataset_name,
                                              tokenizer=tokenizer,
                                              batch_size=batch_size,
                                              text_infilling=True,
                                              max_length=max_length,
                                              task=task)
    
    # step3 wrap model to peft model
    model_config_dict = model.config.to_dict()
    if "gpt" in model_name:
        token_dim = model_config_dict['n_embd']
        num_attention_heads= model_config_dict['n_head']
        num_layers = model_config_dict['n_layer']
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
    lit_model = Lit_RobertaMLM_text_infilling(peft_model,
                                              tokenizer=tokenizer,
                                              lr =lr,
                                              optimizer=optimizer,
                                              dataset = dataset_name,
                                              text_label_dict=TEXT_LABEL_DICT,
                                              token_num=num_tokens)
    
    trainer = Trainer(max_epochs=n_epochs,accelerator="gpu",enable_progress_bar=True,fast_dev_run=False)

    # step5 eval,train,eval
    pre_pt_acc=trainer.test(lit_model,val_loader)[0]['test_acc']
    trainer.fit(lit_model,train_loader)
    post_pt_acc = trainer.test(lit_model,val_loader)[0]['test_acc']

    # step6 save soft prompt
    output_dir = args["output_dir"]
    file_name = f"{model_name}_{dataset_name}_{seed}_{lr}_{n_epochs}"
    output_dir = os.path.join(output_dir,file_name)
    if args["config_path"]:
        os.makedirs(output_dir,exist_ok=True)
        checkpoint_name = state_file.split("/")[-1].split(".")[0]
        output_name = f"{checkpoint_name}_prompt_state.pth"
        output_dir = os.path.join(output_dir,output_name)
        prompt = lit_model.model.prompt_encoder
        torch.save(prompt.state_dict(),output_dir)
    else:
        lit_model.model.save_pretrained(output_dir)

    # upload result
    res = dict(args)
    res["pre_pt_acc"]=pre_pt_acc
    res["post_pt_acc"]=post_pt_acc

    db_config_path = args["db_config_path"]
    db_config = get_config(db_config_path)
    db = connect(db_config)
    collection_name = args["collection_name"]
    insert_result(db,collection_name,res)
    
if __name__=="__main__":
    args = get_args()
    args = vars(args)
    print(args)
    main(args)