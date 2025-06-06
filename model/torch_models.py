import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
# a wrapper class for any pytorch model

class Lit_Model(pl.LightningModule):
    def __init__(self,model,learning_rate=1e-3,optimizer="SGD",lr_scheduler = "ReduceLROnPlateau", t_max = 100,patience=3,transpose_input=False,num_classes=2):
        super(Lit_Model,self).__init__()
        self.model=model
        self.test_acc = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes)
        self.learning_rate = learning_rate
        self.patience = patience
        self.t_max = t_max
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.transpose_input = transpose_input
    def forward(self,x):
        if self.transpose_input:
            x = x.permute((0,2,3,1))
        return self.model(x)
    
    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate,momentum=0.9, weight_decay=5e-4)
        if self.lr_scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=self.patience)
        elif self.lr_scheduler=="CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max)
        elif self.lr_scheduler == None:
            return optimizer
        return [optimizer],[{"scheduler": scheduler,"monitor":"train_loss"}]
    
    def training_step(self,batch,batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs,labels)
        self.log("train_loss", loss)
        return {"loss":loss}
    
    def validation_step(self,batch,batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs,labels)
        self.valid_acc(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc",self.valid_acc)
        return {"val_loss":loss}

    def test_step(self,batch,batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs,labels)
        self.test_acc(outputs, labels)
        self.log("test_loss", loss)
        self.log('test_acc', self.test_acc)
        return {"test_loss":loss}
    

class Lit_Bert(Lit_Model):
    def __init__(self,model,learning_rate=1e-3,optimizer="SGD",lr_scheduler = "ReduceLROnPlateau", t_max = 100,patience=3,transpose_input=False,num_classes=2):
        super(Lit_Bert,self).__init__(model,learning_rate,optimizer,lr_scheduler,t_max,patience,transpose_input,num_classes)
    def forward(self,input_ids,attention_mask):
        return self.model(input_ids,attention_mask)
    def training_step(self,batch,batch_idx):
        input_ids,attention_mask,labels = batch
        outputs = self(input_ids,attention_mask)
        loss = F.cross_entropy(outputs,labels)
        self.log("train_loss", loss)
        return {"loss":loss}
    def validation_step(self,batch,batch_idx):
        input_ids,attention_mask,labels = batch
        outputs = self(input_ids,attention_mask)
        loss = F.cross_entropy(outputs,labels)
        self.valid_acc(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc",self.valid_acc)
        return {"val_loss":loss}
    def test_step(self,batch,batch_idx):
        input_ids,attention_mask,labels = batch
        outputs = self(input_ids,attention_mask)
        loss = F.cross_entropy(outputs,labels)
        self.test_acc(outputs, labels)
        self.log("test_loss", loss)
        self.log('test_acc', self.test_acc)
        return {"test_loss":loss}

class Lit_RobertaSeqCls(pl.LightningModule):
    def __init__(self,model,num_classes,lr,optimizer="Adam",lr_scheduler = "ReduceLROnPlateau",patience=3):
        super(Lit_RobertaSeqCls,self).__init__()
        self.model=model
        self.test_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
        self.lr=lr
        self.optimizer = optimizer
        self.lr_scheduler=lr_scheduler
        self.patience = 3

    def forward(self,input_ids,attention_mask):
        return self.model(input_ids,attention_mask)
    
    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        if self.lr_scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=self.patience)
        return [optimizer],[{"scheduler": scheduler,"monitor":"train_loss"}]
    
    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids,attention_mask)[0]
        loss = F.cross_entropy(outputs,labels)
        self.log("train_loss", loss)
        return {"loss":loss}
    
    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids,attention_mask)[0]
        loss = F.cross_entropy(outputs,labels)
        self.valid_acc(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc",self.valid_acc)
        return {"val_loss":loss}

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids,attention_mask)[0]
        loss = F.cross_entropy(outputs,labels)
        self.test_acc(outputs, labels)
        self.log("test_loss", loss)
        self.log('test_acc', self.test_acc)
        return {"test_loss":loss}


class Lit_RobertaMLM_text_infilling(pl.LightningModule):
    def __init__(self,model,tokenizer,lr,optimizer="Adam",lr_scheduler = "ReduceLROnPlateau",patience=3,SPA=False,dataset="sst2",text_label_dict=None,token_num=0,
                 enable_dp=False,delta=1e-5,epsilon=1,noise_multiplier=1.0,max_grad_norm=1.0):
        super(Lit_RobertaMLM_text_infilling,self).__init__()
        self.model=model
        #self.tokenizer=tokenizer
        if text_label_dict:
            self.data_label_idx=[]
            text_labels = text_label_dict[dataset]
            for text_label_list in text_labels:
                if tokenizer.name_or_path=="roberta-base":  
                    idx_list=[tokenizer.encode(text_label)[1] for text_label in text_label_list]
                elif tokenizer.name_or_path=="gpt2":  
                    idx_list=[tokenizer.encode(text_label)[0] for text_label in text_label_list]
                self.data_label_idx.append(idx_list)
        else:
            self.data_label_idx=None
            if tokenizer.name_or_path=="gpts":
                _index = 0
            elif tokenizer.name_or_path=="roberta-base":
                _index = 1
            if dataset in ["sst2","imdb"]:
                self.pos_idx = tokenizer.encode(" positive")[_index]
                self.neg_idx = tokenizer.encode(" negative")[_index]
            elif dataset in ["fpb","tweeteval_sentiment"]:
                self.pos_idx = tokenizer.encode(" positive")[_index]
                self.neg_idx = tokenizer.encode(" negative")[_index]
                self.neutral_idx = tokenizer.encode(" moderate")[_index]
            elif dataset in ["mnli","snli"]:
                self.pos_idx = tokenizer.encode(" yes")[_index]
                self.neg_idx = tokenizer.encode(" no")[_index]
                self.neutral_idx = tokenizer.encode(" moderate")[_index]
            elif dataset in ["qnli"]:
                self.pos_idx = tokenizer.encode(" yes")[_index]
                self.neg_idx = tokenizer.encode(" no")[_index]
        if dataset in ["sst2","imdb","qnli"]:
            num_classes = 2
            self.test_acc = torchmetrics.classification.BinaryAccuracy()
            self.valid_acc = torchmetrics.classification.BinaryAccuracy()
        else:
            if dataset in ["fpb","tweeteval_sentiment","mnli","snli"]:
                num_classes =3
            elif dataset in ["agnews"]:
                num_classes = 4
            elif dataset in ["arisetv","trec"]:
                num_classes = 6
            elif dataset in ["dbpedia"]:
                num_classes = 14
            self.test_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
            self.valid_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
    
        self.lr=lr
        self.optimizer = optimizer
        self.lr_scheduler=lr_scheduler
        self.patience = 3
        self.loss_fn = nn.CrossEntropyLoss()
        self.SPA = SPA # whether the model is derived from SPA
        self.dataset=dataset
        self.token_num = token_num #soft prompt token num
        # DP related
        self.enable_dp = enable_dp
        self.delta = delta
        self.epsilon = epsilon
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        if self.enable_dp:
            from opacus import PrivacyEngine
            self.privacy_engine = PrivacyEngine()
    def forward(self,input_ids,attention_mask):
        if self.SPA:
            return self.model(input_ids,attention_mask)
        else:
            return self.model(input_ids=input_ids,attention_mask=attention_mask)
    
    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        elif self.optimizer =="AdamW":
            optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),lr=self.lr)

        if self.enable_dp:
            from opacus.data_loader import DPDataLoader
            assert self.optimizer=="SGD"
            self.trainer.fit_loop.setup_data()
            data_loader = self.trainer.train_dataloader
            if hasattr(self, "dp"):
                self.dp["model"].remove_hooks()
            max_epochs = self.trainer.max_epochs
            dp_model, optimizer, dataloader = self.privacy_engine.make_private_with_epsilon(
                module=self,
                optimizer=optimizer,
                target_delta=self.delta,
                epochs = max_epochs,
                target_epsilon=self.epsilon,
                data_loader=data_loader,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=isinstance(data_loader, DPDataLoader),
            )
            self.dp = {"model": dp_model}
        if self.lr_scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=self.patience)
        return [optimizer],[{"scheduler": scheduler,"monitor":"train_loss"}]
    
    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        ## temperal fix of not correct mask_idx 
        ## todo fix this by adding new argument
        mask_idx = batch['mask_idx']+self.token_num
        if self.SPA:
            logits = self(input_ids=input_ids,attention_mask = attention_mask)
        else:
            output=self(input_ids=input_ids,attention_mask = attention_mask)
            logits = output.logits
        pred_logits=[]
        for logit,idx in zip(logits,mask_idx):
            target_logit = logit[idx]
            if self.data_label_idx:
                pred_logits.append(torch.hstack([target_logit[index].mean() for index in self.data_label_idx])) 
            else:
                if self.dataset in ["sst2","imdb"]:
                    pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.pos_idx]]))
                elif self.dataset in ["fpb","tweeteval_sentiment"]:
                    pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.neutral_idx],target_logit[self.pos_idx]]))
                elif self.dataset in ['mnli',"snli"]:
                    pred_logits.append(torch.hstack([target_logit[self.pos_idx],target_logit[self.neutral_idx],target_logit[self.neg_idx]]))
                elif self.dataset in ["qnli"]:
                    pred_logits.append(torch.hstack([target_logit[self.pos_idx],target_logit[self.neg_idx]]))
        pred_logits=torch.vstack(pred_logits)
        loss = self.loss_fn(pred_logits,label)
        #loss = output.loss
        self.log("train_loss", loss)
        return {"loss":loss}
    
    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label'].to(self.device) # true labels for classification
        ## temperal fix of not correct mask_idx 
        ## todo fix this by adding new argument
        mask_idx = batch['mask_idx']+self.token_num
        if self.SPA:
            logits = self(input_ids=input_ids,attention_mask = attention_mask)
        else:
            output=self(input_ids=input_ids,attention_mask = attention_mask)
            logits = output.logits

        pred_logits=[]
        for logit,idx in zip(logits,mask_idx):
            target_logit = logit[idx]
            if self.data_label_idx:
                pred_logits.append(torch.hstack([target_logit[index].mean() for index in self.data_label_idx])) 
            else:
                if self.dataset in ["sst2","imdb"]:
                    pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.pos_idx]]))
                elif self.dataset in ["fpb","tweeteval_sentiment"]:
                    pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.neutral_idx],target_logit[self.pos_idx]]))
                elif self.dataset in ['mnli',"snli"]:
                    pred_logits.append(torch.hstack([target_logit[self.pos_idx],target_logit[self.neutral_idx],target_logit[self.neg_idx]]))
                elif self.dataset in ["qnli"]:
                    pred_logits.append(torch.hstack([target_logit[self.pos_idx],target_logit[self.neg_idx]]))
            
        pred_logits=torch.vstack(pred_logits)
        pred = torch.argmax(pred_logits,axis=1)
        loss = self.loss_fn(pred_logits,label)
        self.valid_acc(pred, label)
        self.log("val_loss", loss)
        self.log("val_acc",self.valid_acc)
        return {"val_loss":loss}

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label'].to(self.device) # true labels for classification

        ## temperal fix of not correct mask_idx 
        ## todo fix this by adding new argument
        mask_idx = batch['mask_idx']+self.token_num
        if self.SPA:
            logits = self(input_ids=input_ids,attention_mask = attention_mask)
        else:
            output=self(input_ids=input_ids,attention_mask = attention_mask)
            logits = output.logits
        pred_logits=[]
        for logit,idx in zip(logits,mask_idx):
            target_logit = logit[idx]
            if self.data_label_idx:
                pred_logits.append(torch.hstack([target_logit[index].mean() for index in self.data_label_idx])) 
            else:
                if self.dataset in ["sst2","imdb"]:
                    pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.pos_idx]]))
                elif self.dataset in ["fpb","tweeteval_sentiment"]:
                    pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.neutral_idx],target_logit[self.pos_idx]]))
                elif self.dataset in ['mnli',"snli"]:
                    pred_logits.append(torch.hstack([target_logit[self.pos_idx],target_logit[self.neutral_idx],target_logit[self.neg_idx]]))
                elif self.dataset in ["qnli"]:
                    pred_logits.append(torch.hstack([target_logit[self.pos_idx],target_logit[self.neg_idx]]))
            
        pred_logits=torch.vstack(pred_logits)
        pred = torch.argmax(pred_logits,axis=1)
        loss = self.loss_fn(pred_logits,label)
        self.test_acc(pred, label)
        self.log("test_loss", loss)
        self.log('test_acc', self.test_acc)
        return {"test_loss":loss}

class Lit_RobertaMLM_text_infilling_adapt(Lit_RobertaMLM_text_infilling):
    def __init__(self,
                 model,
                 tokenizer,
                 lr,
                 optimizer="Adam",
                 lr_scheduler = "ReduceLROnPlateau",
                 patience=3,
                 SPA=False,
                 dataset="sst2",
                 soft_prompt=None):
        super(Lit_RobertaMLM_text_infilling_adapt,self).__init__(model=model,
                                                             tokenizer=tokenizer,
                                                             lr=lr,
                                                             optimizer=optimizer,
                                                             lr_scheduler=lr_scheduler,
                                                             patience=patience,
                                                             SPA = SPA,
                                                             dataset=dataset)
        self.soft_prompt = soft_prompt# for testing usingo only given soft prompt instead of from dataset
    def forward(self,input_ids,attention_mask,source_prompt):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,source_prompt=source_prompt)
    
    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        mask_idx = batch['mask_idx']
        if self.soft_prompt is not None:
            source_prompt = self.soft_prompt.to(self.device)
        else:
            source_prompt = batch['source_prompt']

        output=self(input_ids=input_ids,attention_mask = attention_mask,source_prompt=source_prompt)
        logits = output.logits
        pred_logits=[]
        for logit,idx in zip(logits,mask_idx):
            target_logit = logit[idx]
            if self.dataset in ["sst2","imdb"]:
                pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.pos_idx]]))
            elif self.dataset in ["fpb","tweeteval_sentiment"]:
                pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.neutral_idx],target_logit[self.pos_idx]]))
        
        pred_logits=torch.vstack(pred_logits)
        loss = self.loss_fn(pred_logits,label)
        #loss = output.loss
        self.log("train_loss", loss)
        return {"loss":loss}
    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        label = batch['label'].to(self.device) # true labels for classification
        mask_idx = batch['mask_idx']
        if self.soft_prompt is not None:
            source_prompt = self.soft_prompt.to(self.device)
        else:
            source_prompt = batch['source_prompt']

        output=self(input_ids=input_ids,attention_mask = attention_mask,source_prompt=source_prompt)
        logits = output.logits
        pred = []
        pred_logits=[]
        for logit,idx in zip(logits,mask_idx):
            target_logit = logit[idx]
            if self.dataset in ["sst2","imdb"]:
                pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.pos_idx]]))
                pred.append(target_logit[self.pos_idx]>target_logit[self.neg_idx])
            elif self.dataset in ["fpb","tweeteval_sentiment"]:
                pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.neutral_idx],target_logit[self.pos_idx]]))
                pred.append(torch.argmax(torch.hstack([target_logit[self.neg_idx],target_logit[self.neutral_idx],target_logit[self.pos_idx]])).item())
        pred_logits=torch.vstack(pred_logits)
        loss = self.loss_fn(pred_logits,label)
        pred = torch.Tensor(pred).to(self.device)
        self.valid_acc(pred, label)
        self.log("val_loss", loss)
        self.log("val_acc",self.valid_acc)
        return {"val_loss":loss}

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label'].to(self.device) # true labels for classification
        mask_idx = batch['mask_idx']
        if self.soft_prompt is not None:
            source_prompt = self.soft_prompt.to(self.device)
        else:
            source_prompt = batch['source_prompt']

        output=self(input_ids=input_ids,attention_mask = attention_mask,source_prompt=source_prompt)
        logits = output.logits
        pred = []
        pred_logits=[]
        for logit,idx in zip(logits,mask_idx):
            target_logit = logit[idx]
            if self.dataset in ["sst2","imdb"]:
                pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.pos_idx]]))
                pred.append(target_logit[self.pos_idx]>target_logit[self.neg_idx])
            elif self.dataset in ["fpb","tweeteval_sentiment"]:
                pred_logits.append(torch.hstack([target_logit[self.neg_idx],target_logit[self.neutral_idx],target_logit[self.pos_idx]]))
                pred.append(torch.argmax(torch.hstack([target_logit[self.neg_idx],target_logit[self.neutral_idx],target_logit[self.pos_idx]])).item())
        pred_logits=torch.vstack(pred_logits)
        loss = self.loss_fn(pred_logits,label)
        pred = torch.Tensor(pred).to(self.device)
        self.test_acc(pred, label)
        self.log("test_loss", loss)
        self.log('test_acc', self.test_acc)
        return {"test_loss":loss}