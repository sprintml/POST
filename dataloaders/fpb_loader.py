import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


label2id = {"positive": 2, "neutral": 1, "negative": 0}
MAX_LENGTH = 256
def get_data_and_split(root):
    data = pd.read_csv(root, encoding='latin-1', header=None)
    data.columns = ["labels", "text"]
    texts = data["text"].to_list()
    labels = data["labels"].to_list()

    text_train, text_val, labels_train, labels_val = train_test_split(  
        texts, labels, test_size=0.25, random_state=42,   
    )  
    return text_train,text_val,labels_train,labels_val


class FPB_Dataset(Dataset):
    def __init__(self,texts,labels,tokenizer,text_infilling=True,max_length=MAX_LENGTH,task="mlm"):
        self.labels = labels
        self.text_infilling = text_infilling
        if self.text_infilling:
            if task=="mlm":
                texts = [text+"It was <mask>." for text in texts]
            elif task=="clm":
                texts = [text+"It was" for text in texts]
        tokenizer.truncation_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenized_text = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
        if self.text_infilling:
            if task=="mlm":
                self.mask_index = [(torch.Tensor(x)==tokenizer.mask_token_id).nonzero().item() for x in self.tokenized_text['input_ids']]
            elif task=="clm":
                self.mask_index=[]
                for x in self.tokenized_text['input_ids']:
                    #print(x)
                    zero_idx=(torch.Tensor(x)==tokenizer.pad_token_id).nonzero()
                    #print(zero_idx)
                    if zero_idx.shape[0]==0:
                        idx = len(x)-1
                    else:
                        idx = zero_idx[0].item()-1
                    self.mask_index.append(idx)
                
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        input_ids = self.tokenized_text['input_ids'][idx]
        attention_mask = self.tokenized_text['attention_mask'][idx]
        if self.text_infilling:
            mask_idx = self.mask_index[idx]
        label = label2id[self.labels[idx]]
        #return torch.IntTensor(data['input_ids']),torch.IntTensor(data["attention_mask"]),data['labels']
        if self.text_infilling:
            return {'input_ids':torch.IntTensor(input_ids), 'attention_mask':torch.IntTensor(attention_mask), 'label':label,"mask_idx":mask_idx}

        else:
            return {'input_ids':torch.IntTensor(input_ids), 'attention_mask':torch.IntTensor(attention_mask), "mask_idx":0,'label':label}


def get_fpb_loader(root,batch_size=16,tokenizer=None,text_infilling=True,max_length=MAX_LENGTH,task="mlm"):
    text_train,text_val,labels_train,labels_val = get_data_and_split(root)

    train_dataset = FPB_Dataset(text_train,labels_train,tokenizer,text_infilling,max_length=max_length,task=task)
    val_dataset = FPB_Dataset(text_val,labels_val,tokenizer,text_infilling,max_length=max_length,task=task)
    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=8)
    return train_loader,val_loader
