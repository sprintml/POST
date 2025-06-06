import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset

MAX_LENGTH = 128

class SST2_Dataset_SPA(Dataset):
    def __init__(self,tokenizer,split):
        if split=="train":
            dataset = load_dataset('glue', 'sst2', split='train')
        elif split == "val":
            dataset = load_dataset('glue', 'sst2', split='validation')
        elif split == "test":
            dataset = load_dataset('glue', 'sst2', split='test')
        
        dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        self.dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]
        return torch.IntTensor(data['input_ids']),torch.IntTensor(data["attention_mask"]),data['labels']

class IMDB_Dataset_SPA(Dataset):
    def __init__(self,tokenizer,split):
        dataset = load_dataset('imdb')
        if split=="train":
            dataset = dataset['train']
        elif split=="val":
            dataset = dataset['test']
        tokenizer.truncation_side = "left"
        dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        self.dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]
        return torch.IntTensor(data['input_ids']),torch.IntTensor(data["attention_mask"]),data['labels']

def get_spa_sst2_loader(batch_size=16,tokenizer=None):
    train_dataset = SST2_Dataset_SPA(tokenizer,"train")
    val_dataset = SST2_Dataset_SPA(tokenizer,"val")
    #test_dataset = SST2_Dataset(tokenizer,"test") # test dataset has -1 as labels
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=8)
    #test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,val_loader

# ood dataset of SST2, take premise of ax dataset
class SST2_OOD_dataset(Dataset):
    def __init__(self,tokenizer):
        dataset = load_dataset('glue','ax',split='test')
        dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        self.dataset = dataset.map(lambda e: tokenizer(e['premise'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]
        return torch.IntTensor(data['input_ids']),torch.IntTensor(data["attention_mask"]),data['labels']
    
def get_spa_sst2_ood_loader(batch_size,tokenizer):
    ood_dataset = SST2_OOD_dataset(tokenizer)
    ood_loader = DataLoader(ood_dataset,batch_size=batch_size,shuffle=False)
    return ood_loader

def get_spa_imdb_loader(batch_size=16,tokenizer=None):
    train_dataset = IMDB_Dataset_SPA(tokenizer,"train")
    val_dataset = IMDB_Dataset_SPA(tokenizer,"val")
    #test_dataset = SST2_Dataset(tokenizer,"test") # test dataset has -1 as labels
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=8)
    #test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,val_loader
    