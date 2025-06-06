import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from collections import defaultdict
import torchmetrics

def constant_factory(value):
    return lambda: value

class Lit_RobertaMLM_text_infilling_multi_prompt_adaptor(pl.LightningModule):
    def __init__(self,model,datasets,tokenizer,text_label_dict,lr,optimizer="Adam",lr_scheduler = "ReduceLROnPlateau",patience=3,soft_prompt=None):
        super(Lit_RobertaMLM_text_infilling_multi_prompt_adaptor,self).__init__()
        self.model=model
        #self.tokenizer=tokenizer
        self.data_label_idx={}
        self.acc_fn = {}
        for dataset_name in datasets:
            self.data_label_idx[dataset_name] = []
            text_labels = text_label_dict[dataset_name]
            # each label has a list of text labels
            for text_label_list in text_labels:
                idx_list=[tokenizer.encode(text_label)[1] for text_label in text_label_list]
                self.data_label_idx[dataset_name].append(idx_list)
            # add acc
            num_classes = len(text_labels)
            if num_classes==2:
                self.acc_fn[dataset_name]=torchmetrics.classification.BinaryAccuracy()
            else:
                self.acc_fn[dataset_name]=torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        print(self.data_label_idx)
        self.lr=lr
        self.optimizer = optimizer
        self.lr_scheduler=lr_scheduler
        self.patience = patience
        self.loss_fn = nn.CrossEntropyLoss()
        self.datasets=datasets
        self.soft_prompt=soft_prompt

    def forward(self,input_ids,attention_mask,source_prompt):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,source_prompt=source_prompt)
    
    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        elif self.optimizer =="AdamW":
            optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),lr=self.lr)

        if self.lr_scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=self.patience)
        return [optimizer],[{"scheduler": scheduler,"monitor":"train_loss"}]
    
    def training_step(self,batch,batch_idx):
        # let'see if we need this
        loss_dict = {}
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        mask_idx = batch['mask_idx']
        datasets = batch["dataset"]
        if self.soft_prompt is not None:
            source_prompt = self.soft_prompt.to(self.device)
        else:
            source_prompt = batch['source_prompt']
        output=self(input_ids=input_ids,attention_mask = attention_mask,source_prompt=source_prompt)
        logits = output.logits

        pred_logits_dict=defaultdict(list)
        dataset_label = defaultdict(list)
        for logit,idx,dataset_name,lab in zip(logits,mask_idx,datasets,label):
            target_logit = logit[idx]

            pred_logit_list = [target_logit[index].mean() for index in self.data_label_idx[dataset_name]] # take the logit with max probability
            pred_logits_dict[dataset_name].append(torch.hstack(pred_logit_list))
            dataset_label[dataset_name].append(lab)
        for dataset_name in pred_logits_dict:
            if len(pred_logits_dict[dataset_name])==1:
                pred_logits = pred_logits_dict[dataset_name][0].unsqueeze(0)
            else:
                pred_logits = torch.vstack(pred_logits_dict[dataset_name])
            loss_dict[dataset_name] = self.loss_fn(pred_logits,torch.LongTensor(dataset_label[dataset_name]).to(self.device))
        #loss = output.loss
        loss = sum(loss_dict.values())
        self.log("train_loss", loss)
        return {"loss":loss}
    
    def validation_step(self,batch,batch_idx):
        loss_dict = {}
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        mask_idx = batch['mask_idx']
        datasets = batch["dataset"]
        if self.soft_prompt is not None:
            source_prompt = self.soft_prompt.to(self.device)
        else:
            source_prompt = batch['source_prompt']
        output=self(input_ids=input_ids,attention_mask = attention_mask,source_prompt=source_prompt)
        logits = output.logits

        pred_logits_dict=defaultdict(list)
        dataset_label = defaultdict(list)
        for logit,idx,dataset_name,lab in zip(logits,mask_idx,datasets,label):
            target_logit = logit[idx]
            pred_logit_list = [target_logit[index].mean() for index in self.data_label_idx[dataset_name]] # take the logit with max probability
            pred_logits_dict[dataset_name].append(torch.hstack(pred_logit_list))
            dataset_label[dataset_name].append(lab)
        for dataset_name in pred_logits_dict:
            if len(pred_logits_dict[dataset_name])==1:
                pred_logits = pred_logits_dict[dataset_name][0].unsqueeze(0)
            else:
                pred_logits = torch.vstack(pred_logits_dict[dataset_name])
            loss_dict[dataset_name] = self.loss_fn(pred_logits,torch.LongTensor(dataset_label[dataset_name]).to(self.device))
            pred = torch.argmax(pred_logits,axis=1)
            acc_=self.acc_fn[dataset_name](pred.to("cpu"),torch.LongTensor(dataset_label[dataset_name]).to("cpu"))
            self.log(f"val_acc_{dataset_name}",acc_)

        #loss = output.loss
        loss = sum(loss_dict.values())
        self.log("val_loss", loss)
        return {"val_loss":loss}
    
    def test_step(self,batch,batch_idx):
        loss_dict = {}
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        mask_idx = batch['mask_idx']
        datasets = batch["dataset"]
        if self.soft_prompt is not None:
            source_prompt = self.soft_prompt.to(self.device)
        else:
            source_prompt = batch['source_prompt']
        output=self(input_ids=input_ids,attention_mask = attention_mask,source_prompt=source_prompt)
        logits = output.logits

        pred_logits_dict=defaultdict(list)
        dataset_label = defaultdict(list)
        for logit,idx,dataset_name,lab in zip(logits,mask_idx,datasets,label):
            target_logit = logit[idx]
            pred_logit_list = [target_logit[index].mean() for index in self.data_label_idx[dataset_name]] # take the logit with max probability
            pred_logits_dict[dataset_name].append(torch.hstack(pred_logit_list))
            dataset_label[dataset_name].append(lab)
        for dataset_name in pred_logits_dict:
            if len(pred_logits_dict[dataset_name])==1:
                pred_logits = pred_logits_dict[dataset_name][0].unsqueeze(0)
            else:
                pred_logits = torch.vstack(pred_logits_dict[dataset_name])
            loss_dict[dataset_name] = self.loss_fn(pred_logits,torch.LongTensor(dataset_label[dataset_name]).to(self.device))
            pred = torch.argmax(pred_logits,axis=1)
            acc_=self.acc_fn[dataset_name](pred.to("cpu"),torch.LongTensor(dataset_label[dataset_name]).to("cpu"))
            self.log(f"acc_{dataset_name}",acc_)
        #loss = output.loss
        loss = sum(loss_dict.values())
        self.log("test_loss", loss)
        
        return {"test_loss":loss}
