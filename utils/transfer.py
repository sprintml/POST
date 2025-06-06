import torch
import torch.nn as nn
from scripts.constants import TEXT_LABEL_DICT
from copy import deepcopy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
# utils for transfer prompt

def evaluate(model,dataset_name,tokenizer,dataloader,n_soft,task):
    model=model.eval()
    device = "cuda"
    data_label_idx=[]
    text_labels = TEXT_LABEL_DICT[dataset_name]
    for text_label_list in text_labels:
        if task=="mlm":
            idx_list=[tokenizer.encode(text_label)[1] for text_label in text_label_list]
            data_label_idx.append(idx_list)
        else:
            idx_list=[tokenizer.encode(text_label)[0] for text_label in text_label_list]
            data_label_idx.append(idx_list)
    cor=0
    tot=0
    for data in dataloader:
        x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}
        mask_idxs = data['mask_idx']+n_soft
        label = data['label']
        logits=model(**x).logits
        pred_logits=[]
        for logit,idx in zip(logits,mask_idxs):
            target_logit = logit[idx]   
            pred_logits.append(torch.hstack([target_logit[index].mean() for index in data_label_idx])) 
        pred_logits=torch.vstack(pred_logits)
        pred = torch.argmax(pred_logits,axis=1).cpu()
        cor+=(pred==label).sum().item()
        tot+=len(label)
    print(f"acc:{cor/tot},({cor}/{tot})") 
    return cor/tot


def absolute_transfer(source_peft_model,target_peft_model,tokenizer,
                      dataloader,target_loader,target_dataset_name,anchor_dataset_name,
                      n_steps,method="embedding",feature_token="mask",num_tokens=100,lr=5e-4,interval=30,loss_name="MSE",prefix=None,task="mlm",
                      accumulate_step=4,lr_scheduler=False,optimizer="Adam"):
    source_peft_model=source_peft_model
    target_peft_model=target_peft_model
    step=0
    n_epochs = n_steps//len(dataloader)+1
    device="cuda"
    if loss_name=="MSE":
        loss_fn = nn.MSELoss()
    elif loss_name=="KLDiv":
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        temperature=2.0
    elif loss_name=="Cos":
        loss_fn = nn.CosineEmbeddingLoss(reduction="mean")
    if optimizer=="Adam":
        optimizer = torch.optim.Adam(target_peft_model.parameters(),lr=lr)
    elif optimizer=="SGD":
        optimizer = torch.optim.SGD(target_peft_model.parameters(),lr=lr)
    if lr_scheduler:
        scheduler = MultiStepLR(optimizer,milestones=[1500,3000,4500,6000,7500],gamma=0.2)
    loss_dict={}
    acc_dict = {}
    loss_list = []
    acc_list = []
    #acc_list_pub=[]
    best_acc=0
    best_prompt=None
    for i in range(n_epochs):
        for data in dataloader:
            step+=1
            if step>n_steps+1:
                break
            x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}
            mask_idx = data['mask_idx']
            if method=="embedding":
            # get embeddings of model with prompt
                source_pt_emb = source_peft_model(**x,output_hidden_states=True).hidden_states[-1]
                target_pt_emb = target_peft_model(**x,output_hidden_states=True).hidden_states[-1]
            
                # take the embedding of the specific token <cls> or <mask>
                if feature_token=="cls":
                    source_pt_emb = source_pt_emb[:,num_tokens,:]
                    target_pt_emb = target_pt_emb[:,num_tokens,:] 

                else:
                    source_pt_emb = torch.vstack([source_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                    target_pt_emb = torch.vstack([target_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                if loss_name=="MSE":
                    loss = loss_fn(target_pt_emb,source_pt_emb)
                elif loss_name=="KLDiv":
                    loss = loss_fn(
                        nn.functional.log_softmax(target_pt_emb / temperature, dim=-1),
                        nn.functional.softmax(source_pt_emb / temperature, dim=-1),)* (temperature) ** 2
                elif loss_name=="Cos":
                    pass
            elif method=="output":
                source_pt_output = source_peft_model(**x).logits
                target_pt_output = target_peft_model(**x).logits
                source_pt_output = torch.vstack([source_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                target_pt_output = torch.vstack([target_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                # take the mask token
                if loss_name=="MSE":
                    loss = loss_fn(target_pt_output,source_pt_output)
                elif loss_name=="KLDiv":
                    loss = loss_fn(
                        nn.functional.log_softmax(target_pt_output / temperature, dim=-1),
                        nn.functional.softmax(source_pt_output / temperature, dim=-1),)* (temperature) ** 2
                elif loss_name=="Cos":
                    pass
            loss = loss/accumulate_step
            loss.backward()
            if step%accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler:
                scheduler.step()
            loss_list.append(loss.detach().cpu().item())
            if step%interval==1:
                print("step:",step,sum(loss_list)/len(loss_list))
                acc=evaluate(target_peft_model,target_dataset_name,tokenizer,target_loader,n_soft=num_tokens,task=task)
                #acc_pub = evaluate(target_peft_model,anchor_dataset_name,tokenizer,dataloader,n_soft=num_tokens)
                acc_list.append(acc)
                acc_dict[str(step)] = acc
                loss_dict[str(step)] = loss.detach().cpu().item()
                #acc_list_pub.append(acc_pub)
                if acc>best_acc:
                    best_acc = acc
                    best_step = step
                    best_prompt = deepcopy(target_peft_model.prompt_encoder)
    last_acc = acc
    steps = list(range(len(acc_list)))
    steps = [step*interval for step in steps]
    plt.plot(steps,acc_list)
    #plt.plot(steps,acc_list_pub)
    if prefix:
        save_name = f"{prefix}_{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    else:
        save_name = f"{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    plt.savefig(save_name)

    return best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict

def absolute_ensemble_transfer(source_peft_model_list,target_peft_model,tokenizer,
                      dataloader,target_loader,target_dataset_name,anchor_dataset_name,
                      n_steps,method="embedding",feature_token="mask",num_tokens=100,lr=5e-4,interval=30,loss_name="MSE",prefix=None,task="mlm",
                      accumulate_step=4,lr_scheduler=False):
    source_peft_model_list=source_peft_model_list
    target_peft_model=target_peft_model
    step=0
    n_epochs = n_steps//len(dataloader)+1
    device="cuda"
    if loss_name=="MSE":
        loss_fn = nn.MSELoss()
    elif loss_name=="KLDiv":
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        temperature=2.0
    elif loss_name=="Cos":
        loss_fn = nn.CosineEmbeddingLoss(reduction="mean")

    optimizer = torch.optim.AdamW(target_peft_model.parameters(),lr=lr)
    if lr_scheduler:
        scheduler = MultiStepLR(optimizer,milestones=[3000,6000,9000,12000,15000],gamma=0.2)
    loss_dict={}
    acc_dict = {}
    loss_list = []
    acc_list = []
    #acc_list_pub=[]
    best_acc=0
    best_prompt=None
    for i in range(n_epochs):
        for data in dataloader:
            step+=1
            if step>n_steps+1:
                break
            x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}
            mask_idx = data['mask_idx']
            total_loss = 0
            if method=="ensemble_embedding":
            # get embeddings of model with prompt
                target_pt_emb = target_peft_model(**x,output_hidden_states=True).hidden_states[-1]
                for source_peft_model in source_peft_model_list:
                    source_pt_emb = source_peft_model(**x,output_hidden_states=True).hidden_states[-1]
                    # take the embedding of the specific token <cls> or <mask>
                    if feature_token=="cls":
                        source_pt_emb = source_pt_emb[:,num_tokens,:]
                        target_pt_emb = target_pt_emb[:,num_tokens,:] 

                    else:
                        source_pt_emb = torch.vstack([source_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                        target_pt_emb = torch.vstack([target_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                    if loss_name=="MSE":
                        total_loss += loss_fn(target_pt_emb,source_pt_emb)
                    elif loss_name=="KLDiv":
                        total_loss += loss_fn(
                            nn.functional.log_softmax(target_pt_emb / temperature, dim=-1),
                            nn.functional.softmax(source_pt_emb / temperature, dim=-1),)* (temperature) ** 2
                    elif loss_name=="Cos":
                        pass

            elif method=="ensemble_output":
                target_pt_output = target_peft_model(**x).logits
                target_pt_output = torch.vstack([target_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                for source_peft_model in source_peft_model_list:
                    source_pt_output = source_peft_model(**x).logits
                    source_pt_output = torch.vstack([source_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                    # take the mask token
                    if loss_name=="MSE":
                        total_loss += loss_fn(target_pt_output,source_pt_output)
                    elif loss_name=="KLDiv":
                        total_loss += loss_fn(
                            nn.functional.log_softmax(target_pt_output / temperature, dim=-1),
                            nn.functional.softmax(source_pt_output / temperature, dim=-1),)* (temperature) ** 2
                    elif loss_name=="Cos":
                        pass
            total_loss = total_loss/3
            total_loss = total_loss/accumulate_step
            total_loss.backward()
            if step%accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler:
                scheduler.step()
            loss_list.append(total_loss.detach().cpu().item())
            if step%interval==1:
                print("step:",step,sum(loss_list)/len(loss_list))
                acc=evaluate(target_peft_model,target_dataset_name,tokenizer,target_loader,n_soft=num_tokens,task=task)
                #acc_pub = evaluate(target_peft_model,anchor_dataset_name,tokenizer,dataloader,n_soft=num_tokens)
                acc_list.append(acc)
                acc_dict[str(step)] = acc
                loss_dict[str(step)] = total_loss.detach().cpu().item()
                #acc_list_pub.append(acc_pub)
                if acc>best_acc:
                    best_acc = acc
                    best_step = step
                    best_prompt = deepcopy(target_peft_model.prompt_encoder)
    last_acc = acc
    steps = list(range(len(acc_list)))
    steps = [step*interval for step in steps]
    plt.plot(steps,acc_list)
    #plt.plot(steps,acc_list_pub)
    if prefix:
        save_name = f"{prefix}_{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    else:
        save_name = f"{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    plt.savefig(save_name)

    return best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict


def absolute_shift_transfer(source_model,target_model,source_peft_model,target_peft_model,tokenizer,
                      dataloader,target_loader,target_dataset_name,anchor_dataset_name,
                      n_steps,method="embedding_shift",feature_token="mask",num_tokens=100,lr=5e-4,interval=30,loss_name="MSE",prefix=None,task="mlm",
                      accumulate_step=4,lr_scheduler=False,optimizer="Adam"):
    source_model = source_model
    target_model = target_model
    source_peft_model=source_peft_model
    target_peft_model=target_peft_model
    step=0
    n_epochs = n_steps//len(dataloader)+1
    device="cuda"
    if loss_name=="MSE":
        loss_fn = nn.MSELoss()
    elif loss_name=="KLDiv":
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        temperature=2.0
    elif loss_name=="Cos":
        loss_fn = nn.CosineEmbeddingLoss(reduction="mean")
    if optimizer=="Adam":
        optimizer = torch.optim.Adam(target_peft_model.parameters(),lr=lr)
    elif optimizer=="SGD":
        optimizer = torch.optim.SGD(target_peft_model.parameters(),lr=lr)
    if lr_scheduler:
        scheduler = MultiStepLR(optimizer,milestones=[1500,3000,4500,6000,7500],gamma=0.2)
    loss_dict={}
    acc_dict = {}
    loss_list = []
    acc_list = []
    best_acc=0
    best_prompt=None
    for i in range(n_epochs):
        for data in dataloader:
            step+=1
            if step>n_steps+1:
                break
            x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}
            mask_idx = data['mask_idx']
            if method=="embedding_shift":
            # get embeddings of model with prompt
                source_emb=source_model(**x,output_hidden_states=True).hidden_states[-1]
                target_emb=target_model(**x,output_hidden_states=True).hidden_states[-1]
                source_pt_emb = source_peft_model(**x,output_hidden_states=True).hidden_states[-1]
                target_pt_emb = target_peft_model(**x,output_hidden_states=True).hidden_states[-1]
            
                # take the embedding of the specific token <cls> or <mask>
                if feature_token=="cls":
                    source_emb = source_emb[:,0,:]
                    target_emb = target_emb[:,0,:]
                    source_pt_emb = source_pt_emb[:,num_tokens,:]
                    target_pt_emb = target_pt_emb[:,num_tokens,:] 

                else:
                    source_emb = torch.vstack([source_emb[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                    target_emb = torch.vstack([target_emb[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                    source_pt_emb = torch.vstack([source_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                    target_pt_emb = torch.vstack([target_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                target_diff = target_pt_emb-target_emb
                source_diff = source_pt_emb-source_emb
                if loss_name=="MSE":
                    loss = loss_fn(target_diff,source_diff)
                elif loss_name=="KLDiv":
                    loss = loss_fn(
                        nn.functional.log_softmax(target_diff / temperature, dim=-1),
                        nn.functional.softmax(source_diff/ temperature, dim=-1),)* (temperature) ** 2
                elif loss_name=="Cos":
                    pass
            elif method=="output_shift":
                source_output=source_model(**x).logits
                target_output=target_model(**x).logits
                source_pt_output = source_peft_model(**x).logits
                target_pt_output = target_peft_model(**x).logits
                source_output = torch.vstack([source_output[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                target_output = torch.vstack([target_output[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                source_pt_output = torch.vstack([source_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                target_pt_output = torch.vstack([target_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                # take the mask token
                target_diff = target_pt_output-target_output
                source_diff = source_pt_output-source_output
                if loss_name=="MSE":
                    loss = loss_fn(target_diff,source_diff)
                elif loss_name=="KLDiv":
                    loss = loss_fn(
                        nn.functional.log_softmax(target_diff / temperature, dim=-1),
                        nn.functional.softmax(source_diff / temperature, dim=-1),)* (temperature) ** 2
                elif loss_name=="Cos":
                    pass
            loss = loss/accumulate_step
            loss.backward()
            if step%accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler:
                scheduler.step()
            loss_list.append(loss.detach().cpu().item())
            if step%interval==1:
                print("step:",step,sum(loss_list)/len(loss_list))
                acc=evaluate(target_peft_model,target_dataset_name,tokenizer,target_loader,n_soft=num_tokens,task=task)
                #acc_pub = evaluate(target_peft_model,anchor_dataset_name,tokenizer,dataloader,n_soft=num_tokens)
                acc_list.append(acc)
                acc_dict[str(step)] = acc
                loss_dict[str(step)] = loss.detach().cpu().item()
                #acc_list_pub.append(acc_pub)
                if acc>best_acc:
                    best_acc = acc
                    best_step = step
                    best_prompt = deepcopy(target_peft_model.prompt_encoder)
    last_acc = acc
    steps = list(range(len(acc_list)))
    steps = [step*interval for step in steps]
    plt.plot(steps,acc_list)
    #plt.plot(steps,acc_list_pub)
    if prefix:
        save_name = f"{prefix}_{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    else:
        save_name = f"{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    plt.savefig(save_name)

    return best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict

def mix_transfer(source_model,target_model,source_peft_model,target_peft_model,tokenizer,
                 dataloader,target_loader,target_dataset_name,anchor_dataset_name,
                 n_steps,method="mix_transfer",feature_token="mask",num_tokens=100,lr=5e-4,interval=30,loss_name="MSE",prefix=None,task="mlm",
                accumulate_step=1,lr_scheduler=False,optimizer="Adam",alpha=0.5,batch_num=0):
    source_model = source_model
    target_model = target_model
    source_peft_model=source_peft_model
    target_peft_model=target_peft_model
    step=0
    if batch_num>0:
        n_epochs = n_steps//batch_num
    else:
        n_epochs = n_steps//len(dataloader)+1
    device="cuda"
    if loss_name=="MSE":
        loss_fn = nn.MSELoss()
    elif loss_name=="KLDiv":
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        temperature=2.0
    elif loss_name=="Cos":
        loss_fn = nn.CosineEmbeddingLoss(reduction="mean")
    if optimizer=="Adam":
        optimizer = torch.optim.Adam(target_peft_model.parameters(),lr=lr)
    elif optimizer=="SGD":
        optimizer = torch.optim.SGD(target_peft_model.parameters(),lr=lr)
    if lr_scheduler:
        scheduler = MultiStepLR(optimizer,milestones=[1500,3000,4500,6000,7500],gamma=0.2)
    loss_dict={}
    acc_dict = {}
    loss_list = []
    acc_list = []
    best_acc=0
    best_prompt=None
    for i in range(n_epochs):
        for count, data in enumerate(dataloader):
            step+=1
            if step>n_steps+1:
                break
            if batch_num>0 and count+1 > batch_num: # if use different num of samples
                break

            x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}
            mask_idx = data['mask_idx']

            source_output=source_model(**x).logits
            target_output=target_model(**x).logits
            source_pt_output = source_peft_model(**x).logits
            target_pt_output = target_peft_model(**x).logits
            source_output = torch.vstack([source_output[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
            target_output = torch.vstack([target_output[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
            source_pt_output = torch.vstack([source_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
            target_pt_output = torch.vstack([target_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])

            # take the mask token
            target_diff = target_pt_output-target_output
            source_diff = source_pt_output-source_output
            if loss_name=="MSE":
                loss1 = loss_fn(target_diff,source_diff)
                loss2 = loss_fn(target_pt_output,source_pt_output)
            elif loss_name=="KLDiv":
                loss1 = loss_fn(
                    nn.functional.log_softmax(target_diff / temperature, dim=-1),
                    nn.functional.softmax(source_diff / temperature, dim=-1),)* (temperature) ** 2
                loss2 = loss_fn(
                    nn.functional.log_softmax(target_pt_output / temperature, dim=-1),
                    nn.functional.softmax(source_pt_output / temperature, dim=-1),)* (temperature) ** 2
            elif loss_name=="Cos":
                pass
            
            loss= alpha*loss1+ (1-alpha)*loss2
            
            loss = loss/accumulate_step
            loss.backward()
            if step%accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler:
                scheduler.step()
            loss_list.append(loss.detach().cpu().item())
            if step%interval==1:
                print("step:",step,sum(loss_list)/len(loss_list))
                acc=evaluate(target_peft_model,target_dataset_name,tokenizer,target_loader,n_soft=num_tokens,task=task)
                #acc_pub = evaluate(target_peft_model,anchor_dataset_name,tokenizer,dataloader,n_soft=num_tokens)
                acc_list.append(acc)
                acc_dict[str(step)] = acc
                loss_dict[str(step)] = loss.detach().cpu().item()
                #acc_list_pub.append(acc_pub)
                if acc>best_acc:
                    best_acc = acc
                    best_step = step
                    best_prompt = deepcopy(target_peft_model.prompt_encoder)
    last_acc = acc
    steps = list(range(len(acc_list)))
    steps = [step*interval for step in steps]
    plt.plot(steps,acc_list)
    #plt.plot(steps,acc_list_pub)
    if prefix:
        save_name = f"{prefix}_{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    else:
        save_name = f"{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    plt.savefig(save_name)

    return best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict



def topk_transfer(source_model,target_model,source_peft_model,target_peft_model,tokenizer,
                      dataloader,target_loader,target_dataset_name,anchor_dataset_name,
                      n_steps,method="topk_transfer",feature_token="mask",num_tokens=100,lr=5e-4,interval=30,loss_name="MSE",prefix=None,task="mlm",
                      accumulate_step=4,lr_scheduler=False,optimizer="Adam",topk=10):
    assert loss_name=="MSE"
    source_model = source_model
    target_model = target_model
    source_peft_model=source_peft_model
    target_peft_model=target_peft_model
    step=0
    n_epochs = n_steps//len(dataloader)+1
    device="cuda"
    if loss_name=="MSE":
        loss_fn = nn.MSELoss()
    elif loss_name=="KLDiv":
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        temperature=2.0
    elif loss_name=="Cos":
        loss_fn = nn.CosineEmbeddingLoss(reduction="mean")
    if optimizer=="Adam":
        optimizer = torch.optim.Adam(target_peft_model.parameters(),lr=lr)
    elif optimizer=="SGD":
        optimizer = torch.optim.SGD(target_peft_model.parameters(),lr=lr)
    if lr_scheduler:
        scheduler = MultiStepLR(optimizer,milestones=[1500,3000,4500,6000,7500],gamma=0.2)
    loss_dict={}
    acc_dict = {}
    loss_list = []
    acc_list = []
    best_acc=0
    best_prompt=None
    for i in range(n_epochs):
        for data in dataloader:
            step+=1
            if step>n_steps+1:
                break
            x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}

            mask_idx = data['mask_idx']
            source_output=source_model(**x).logits
            target_output=target_model(**x).logits
            source_pt_output = source_peft_model(**x).logits
            target_pt_output = target_peft_model(**x).logits
            source_output = torch.vstack([source_output[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
            target_output = torch.vstack([target_output[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
            source_pt_output = torch.vstack([source_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
            target_pt_output = torch.vstack([target_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
            # take the mask token
            target_diff = target_pt_output-target_output
            source_diff = source_pt_output-source_output
        

            _,top_idx = torch.topk(source_diff,k=topk) # only optimize topk difference
            target_diff = target_diff.gather(dim=1,index=top_idx)
            source_diff = source_diff.gather(dim=1,index=top_idx)

            loss = loss_fn(target_diff,source_diff)
            loss = loss/accumulate_step
            loss.backward()
            if step%accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler:
                scheduler.step()
            loss_list.append(loss.detach().cpu().item())
            if step%interval==1:
                print("step:",step,sum(loss_list)/len(loss_list))
                acc=evaluate(target_peft_model,target_dataset_name,tokenizer,target_loader,n_soft=num_tokens,task=task)
                #acc_pub = evaluate(target_peft_model,anchor_dataset_name,tokenizer,dataloader,n_soft=num_tokens)
                acc_list.append(acc)
                acc_dict[str(step)] = acc
                loss_dict[str(step)] = loss.detach().cpu().item()
                #acc_list_pub.append(acc_pub)
                if acc>best_acc:
                    best_acc = acc
                    best_step = step
                    best_prompt = deepcopy(target_peft_model.prompt_encoder)
    last_acc = acc
    steps = list(range(len(acc_list)))
    steps = [step*interval for step in steps]
    plt.plot(steps,acc_list)
    #plt.plot(steps,acc_list_pub)
    if prefix:
        save_name = f"{prefix}_{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    else:
        save_name = f"{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    plt.savefig(save_name)

    return best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict

def inference_transfer(source_model,target_model,source_peft_model,tokenizer,
                      target_loader,target_dataset_name,num_tokens=100,task="mlm"):
    device="cuda"
    data_label_idx=[]
    text_labels = TEXT_LABEL_DICT[target_dataset_name]
    for text_label_list in text_labels:
        if task=="mlm":
            idx_list=[tokenizer.encode(text_label)[1] for text_label in text_label_list]
            data_label_idx.append(idx_list)
        else:
            idx_list=[tokenizer.encode(text_label)[0] for text_label in text_label_list]
            data_label_idx.append(idx_list)

    cor=0
    tot=0
    for data in target_loader:
        x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}
        mask_idx = data['mask_idx']
        label = data['label']
        source_output=source_model(**x).logits
        target_output=target_model(**x).logits
        source_pt_output = source_peft_model(**x).logits

        source_output = torch.vstack([source_output[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
        target_output = torch.vstack([target_output[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
        
        source_pt_output = torch.vstack([source_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
        
        target_pt_logit = target_output+source_pt_output-source_output
        
        pred_logits=[]
        for target_logit in target_pt_logit:
            pred_logits.append(torch.hstack([target_logit[index].mean() for index in data_label_idx])) 
        pred_logits=torch.vstack(pred_logits)
        pred = torch.argmax(pred_logits,axis=1).cpu()
        cor+=(pred==label).sum().item()
        tot+=len(label)
    print(f"acc:{cor/tot},({cor}/{tot})")   
    return cor/tot

def topk_output(source_peft_model,target_peft_model,tokenizer,
                      dataloader,target_loader,target_dataset_name,anchor_dataset_name,
                      n_steps,method="topk_transfer",feature_token="mask",num_tokens=100,lr=5e-4,interval=30,loss_name="MSE",prefix=None,task="mlm",
                      accumulate_step=4,lr_scheduler=False,optimizer="Adam",topk=10):
    assert loss_name=="MSE"
    source_peft_model=source_peft_model
    target_peft_model=target_peft_model
    step=0
    n_epochs = n_steps//len(dataloader)+1
    device="cuda"
    if loss_name=="MSE":
        loss_fn = nn.MSELoss()
    elif loss_name=="KLDiv":
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        temperature=2.0
    elif loss_name=="Cos":
        loss_fn = nn.CosineEmbeddingLoss(reduction="mean")
    if optimizer=="Adam":
        optimizer = torch.optim.Adam(target_peft_model.parameters(),lr=lr)
    elif optimizer=="SGD":
        optimizer = torch.optim.SGD(target_peft_model.parameters(),lr=lr)
    if lr_scheduler:
        scheduler = MultiStepLR(optimizer,milestones=[1500,3000,4500,6000,7500],gamma=0.2)
    loss_dict={}
    acc_dict = {}
    loss_list = []
    acc_list = []
    best_acc=0
    best_prompt=None
    for i in range(n_epochs):
        for data in dataloader:
            step+=1
            if step>n_steps+1:
                break
            x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}

            mask_idx = data['mask_idx']
            source_pt_output = source_peft_model(**x).logits
            target_pt_output = target_peft_model(**x).logits

            source_pt_output = torch.vstack([source_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
            target_pt_output = torch.vstack([target_pt_output[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])

        
            _,top_idx = torch.topk(source_pt_output,k=topk) # only optimize topk difference
            source_pt_output = source_pt_output.gather(dim=1,index=top_idx)
            target_pt_output = target_pt_output.gather(dim=1,index=top_idx)

            loss = loss_fn(target_pt_output,source_pt_output)
            loss = loss/accumulate_step
            loss.backward()
            if step%accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler:
                scheduler.step()
            loss_list.append(loss.detach().cpu().item())
            if step%interval==1:
                print("step:",step,sum(loss_list)/len(loss_list))
                acc=evaluate(target_peft_model,target_dataset_name,tokenizer,target_loader,n_soft=num_tokens,task=task)
                #acc_pub = evaluate(target_peft_model,anchor_dataset_name,tokenizer,dataloader,n_soft=num_tokens)
                acc_list.append(acc)
                acc_dict[str(step)] = acc
                loss_dict[str(step)] = loss.detach().cpu().item()
                #acc_list_pub.append(acc_pub)
                if acc>best_acc:
                    best_acc = acc
                    best_step = step
                    best_prompt = deepcopy(target_peft_model.prompt_encoder)
    last_acc = acc
    steps = list(range(len(acc_list)))
    steps = [step*interval for step in steps]
    plt.plot(steps,acc_list)
    #plt.plot(steps,acc_list_pub)
    if prefix:
        save_name = f"{prefix}_{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    else:
        save_name = f"{method}_pub_{anchor_dataset_name}_pri_{target_dataset_name}_{loss_name}.png"
    plt.savefig(save_name)

    return best_prompt,best_acc,best_step,last_acc,acc_dict,loss_dict

def relative_transfer(source_model,target_model,source_peft_model,target_peft_model,tokenizer,
                      dataloader,target_loader,target_dataset_name,
                      source_anchors,target_anchors,
                      n_steps,method="embedding",feature_token="mask",num_tokens=100,lr=5e-4,interval=30):
    source_model = source_model
    target_model = target_model
    source_peft_model=source_peft_model
    target_peft_model=target_peft_model
    loss_list=[]
    step=0
    n_epochs = n_steps//len(dataloader)+1
    device="cuda"
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(target_peft_model.parameters(),lr=lr)
    loss_list = []
    best_acc=0
    best_prompt=None
    for i in range(n_epochs):
        for data in dataloader:
            step+=1
            if step>n_steps:
                break
            x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}
            mask_idx = data['mask_idx']
            # get embeddings of model without prompt
            source_emb=source_model(**x,output_hidden_states=True).hidden_states[-1]
            target_emb=target_model(**x,output_hidden_states=True).hidden_states[-1]

            # get embeddings of model with prompt
            source_pt_emb = source_peft_model(**x,output_hidden_states=True).hidden_states[-1]
            target_pt_emb = target_peft_model(**x,output_hidden_states=True).hidden_states[-1]
            
            # take the embedding of the specific token <cls> or <mask>
            if feature_token=="cls":
                source_emb = source_emb[:,0,:]
                target_emb = target_emb[:,0,:]

                source_pt_emb = source_pt_emb[:,num_tokens,:]
                target_pt_emb = target_pt_emb[:,num_tokens,:] 

            else:
                source_emb = torch.vstack([source_emb[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                target_emb = torch.vstack([target_emb[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                source_pt_emb = torch.vstack([source_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                target_pt_emb = torch.vstack([target_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])

            # normalize
            source_emb = F.normalize(source_emb, p=2, dim=-1)
            target_emb = F.normalize(target_emb, p=2, dim=-1)
            rel_source = torch.einsum("bm, am -> ba", source_emb, source_anchors)
            rel_target = torch.einsum("bm, am -> ba", target_emb, target_anchors)

            source_pt_emb =  F.normalize(source_pt_emb, p=2, dim=-1)
            target_pt_emb =  F.normalize(target_pt_emb, p=2, dim=-1)
            rel_pt_source = torch.einsum("bm, am -> ba", source_pt_emb, source_anchors)
            rel_pt_target = torch.einsum("bm, am -> ba", target_pt_emb, target_anchors)
            # project to relative space

            if method == "relative_shift":
                source_diff = rel_pt_source - rel_source
                target_diff = rel_pt_target - rel_target
                loss = loss_fn(target_diff,source_diff)
            elif method == "relative":
                loss = loss_fn(rel_pt_target,rel_pt_source)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.detach().cpu().item())
            if step%interval==1:
                print("step:",step,sum(loss_list)/len(loss_list))
                acc=evaluate(target_peft_model,target_dataset_name,tokenizer,target_loader,n_soft=num_tokens)
                if acc>best_acc:
                    best_acc = acc
                    best_step = step
                    best_prompt = deepcopy(target_peft_model.prompt_encoder)
    last_acc = acc
    return best_prompt,best_acc,best_step,last_acc
    
# train translation without prompt
def translation_transfer(source_model,target_model,source_peft_model,target_peft_model,tokenizer,
                      dataloader,target_loader,target_dataset_name,
                      n_steps,feature_token="mask",num_tokens=100,lr=5e-4,interval=30):
    device="cuda"
    source_model = source_model
    target_model = target_model
    source_peft_model=source_peft_model
    target_peft_model=target_peft_model
    # first train a translator

    source_config_dict = source_model.config.to_dict()
    target_config_dict = target_model.config.to_dict()
    dim_source = source_config_dict['hidden_size']
    dim_target = target_config_dict["hidden_size"]
    assert(dim_source<=dim_target)
    translator = nn.Linear(dim_target,dim_target).to(device) # translate hidden state of source to target
    optimizer_1 = torch.optim.Adam(translator.parameters(),lr=lr)
    loss_fn = torch.nn.MSELoss()
    def transform(emb_source,emb_target,dim_source,dim_target):
        # pad
        sample_num=emb_source.shape[0]
        dim_diff = dim_target-dim_source
        if dim_diff>0:
            pad = torch.zeros(sample_num,dim_diff).to(device)
            emb_target = torch.hstack([emb_target,pad])
        # normalize
        emb_source = F.normalize(emb_source, p=2, dim=-1)
        emb_target = F.normalize(emb_target, p=2, dim=-1)
        return emb_source, emb_target
    
    # training translator for 1 epoch
    loss_list=[]
    step=0
    num_steps=4000
    n_epochs = num_steps//len(dataloader)+1
    for i in range(n_epochs):
        for data in dataloader:
            step+=1
            if step>num_steps:
                break
            x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}
            mask_idx = data['mask_idx']
            source_emb=source_model(**x,output_hidden_states=True).hidden_states[-1]
            target_emb=target_model(**x,output_hidden_states=True).hidden_states[-1]
            if feature_token=="cls":
                source_emb = source_emb[:,0,:]
                target_emb = target_emb[:,0,:]
            elif feature_token=="mask":
                source_emb = torch.vstack([source_emb[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                target_emb = torch.vstack([target_emb[i,j,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
            
            source_emb,target_emb = transform(source_emb,target_emb,dim_source,dim_target)
            pred = translator(source_emb)
            loss = loss_fn(target_emb,pred)
            loss.backward()
            optimizer_1.step()
            optimizer_1.zero_grad()
            loss_list.append(loss.detach().cpu().item())
            if step%400==1:
                print("translator",step,sum(loss_list)/len(loss_list))
            

    # then tune-prompt
    loss_list=[]
    step=0
    n_epochs = n_steps//len(dataloader)+1
    device="cuda"
    optimizer_2 = torch.optim.AdamW(target_peft_model.parameters(),lr=lr)
    best_acc=0
    best_prompt=None
    for i in range(n_epochs):
        for data in dataloader:
            step+=1
            if step>n_steps:
                break
            x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}
            mask_idx = data['mask_idx']
            # get embeddings of model without prompt
            source_pt_emb = source_peft_model(**x,output_hidden_states=True).hidden_states[-1]
            target_pt_emb = target_peft_model(**x,output_hidden_states=True).hidden_states[-1]
            
            # take the embedding of the specific token <cls> or <mask>
            if feature_token=="cls":
                source_pt_emb = source_pt_emb[:,num_tokens,:]
                target_pt_emb = target_pt_emb[:,num_tokens,:] 
            elif feature_token=="mask":
                source_pt_emb = torch.vstack([source_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                target_pt_emb = torch.vstack([target_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
            
            # normalize
            source_pt_emb,target_pt_emb = transform(source_pt_emb,target_pt_emb,dim_source,dim_target)
            pred = translator(source_pt_emb)
            loss = loss_fn(target_pt_emb,pred)
            loss.backward()
            optimizer_2.step()
            optimizer_2.zero_grad()
            loss_list.append(loss.detach().cpu().item())
            if step%interval==1:
                print("step:",step,sum(loss_list)/len(loss_list))
                acc=evaluate(target_peft_model,target_dataset_name,tokenizer,target_loader,n_soft=num_tokens)
                if acc>best_acc:
                    best_acc = acc
                    best_step = step
                    best_prompt = deepcopy(target_peft_model.prompt_encoder)
    last_acc = acc
    return best_prompt,best_acc,best_step,last_acc

# optimize translator and softprompt alternatively
# train tranlation with soft prompt
def translation_after_prompt_transfer(source_peft_model,target_peft_model,tokenizer,
                      dataloader,target_loader,target_dataset_name,
                      n_steps,feature_token="mask",num_tokens=100,lr=5e-4,interval=30):
    # training linear translator is much easier, so should use alternative training to split both
    device="cuda"
    source_peft_model=source_peft_model
    target_peft_model=target_peft_model
    # first train a translator

    source_config_dict = source_peft_model.config.to_dict()
    target_config_dict = target_peft_model.config.to_dict()
    dim_source = source_config_dict['hidden_size']
    dim_target = target_config_dict["hidden_size"]
    assert(dim_source<=dim_target)
    
    translator = nn.Linear(dim_target,dim_target).to(device) # translate hidden state of source to target
    # create optimizer for translator and target_peft_model
    optimizer_1 = torch.optim.Adam(translator.parameters(),lr=lr)
    optimizer_2 = torch.optim.Adam(target_peft_model.parameters(),lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    def transform(emb_source,emb_target,dim_source,dim_target):
        # pad
        sample_num=emb_source.shape[0]
        dim_diff = dim_target-dim_source
        if dim_diff>0:
            pad = torch.zeros(sample_num,dim_diff).to(device)
            emb_target = torch.hstack([emb_target,pad])
        # normalize
        emb_source = F.normalize(emb_source, p=2, dim=-1)
        emb_target = F.normalize(emb_target, p=2, dim=-1)
        return emb_source, emb_target
    #training translator for 1 epoch
    loss_list=[]
    step=0
    n_epochs = n_steps//len(dataloader)+1
    device="cuda"
    best_acc=0
    best_prompt=None
    for i in range(n_epochs):
        for data in dataloader:
            step+=1
            if step>n_steps:
                break
            x = {"input_ids":data['input_ids'].to(device),"attention_mask":data['attention_mask'].to(device)}
            mask_idx = data['mask_idx']
            if step%5==1:
                # optimize the translator
                source_pt_emb=source_peft_model(**x,output_hidden_states=True).hidden_states[-1]
                target_pt_emb=target_peft_model(**x,output_hidden_states=True).hidden_states[-1]
                if feature_token=="cls":
                    source_pt_emb = source_pt_emb[:,num_tokens,:]
                    target_pt_emb = target_pt_emb[:,num_tokens,:]
                elif feature_token=="mask":
                    source_pt_emb = torch.vstack([source_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                    target_pt_emb = torch.vstack([target_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                
                source_pt_emb,target_pt_emb = transform(source_pt_emb,target_pt_emb,dim_source,dim_target)
                # first update translator
                pred = translator(source_pt_emb)
                loss = loss_fn(target_pt_emb,pred)
                loss.backward()
                optimizer_1.step()
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
            # then update soft prompt
            if step>200:
                source_pt_emb=source_peft_model(**x,output_hidden_states=True).hidden_states[-1]
                target_pt_emb=target_peft_model(**x,output_hidden_states=True).hidden_states[-1]
                if feature_token=="cls":
                    source_pt_emb = source_pt_emb[:,num_tokens,:]
                    target_pt_emb = target_pt_emb[:,num_tokens,:]
                elif feature_token=="mask":
                    source_pt_emb = torch.vstack([source_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                    target_pt_emb = torch.vstack([target_pt_emb[i,j+num_tokens,:] for i,j in zip(range(len(mask_idx)),mask_idx)])
                
                source_pt_emb,target_pt_emb = transform(source_pt_emb,target_pt_emb,dim_source,dim_target)
                pred = translator(source_pt_emb)
                loss = loss_fn(target_pt_emb,pred)
                loss.backward()
                optimizer_2.step()
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
    

                loss_list.append(loss.detach().cpu().item())
                if step%interval==1:
                    print("step:",step,sum(loss_list)/len(loss_list))
                    acc=evaluate(target_peft_model,target_dataset_name,tokenizer,target_loader,n_soft=num_tokens)
                    if acc>best_acc:
                        best_acc = acc
                        best_step = step
                        best_prompt = deepcopy(target_peft_model.prompt_encoder)
    last_acc = acc
    return best_prompt,best_acc,best_step,last_acc
