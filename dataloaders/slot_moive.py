import os
import torch
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


MAX_LENGTH=128


class Slot_Movie_Dataset(Dataset):
    def __init__(self,tokenizer,split,max_length=MAX_LENGTH,seed=None,data_path=None,category="director",tag="train"):
        assert data_path is not None    
        if category=="director":
            data_path = os.path.join(data_path,"Director")

        elif category=="genre":
            data_path = os.path.join(data_path,"Genre")
            
        data_files = {"train": "train.csv", "test": "test.csv"}
        dataset = load_dataset(data_path, data_files=data_files)

        if split=="train":
            dataset =  dataset['train']
            if tag=="train":
                dataset = dataset.select(range(int(len(dataset)*0.6)))
            elif tag=="transfer":
                dataset = dataset.select(range(int(len(dataset)*0.6),len(dataset)))
            else:
                raise ValueError("tag should be train or transfer")
        elif split=="test":
            dataset =  dataset['test']
        def process(e):
            if category=="director":
                template_str = "Context:{context}\nThe director of the movie is"
            elif category=="genre":
                template_str = "Answer the genre of the movie directly from the context.\nContext:{context}.\n This movie is"
            e["sentence"] = template_str.format(context=e["content"])
            e["label"] = e['label']
            sentence_ids = tokenizer(e['sentence'],return_tensors="pt").input_ids.squeeze()
            label_ids = tokenizer(e['label'],return_tensors="pt").input_ids.squeeze()
            combined_ids = torch.cat([sentence_ids[:-1], label_ids])
            combined_labels = torch.full_like(combined_ids, -100)  # Initialize with ignored tokens
            combined_labels[-len(label_ids):] = label_ids
            input_ids = torch.full((max_length,), tokenizer.pad_token_id, dtype=torch.long)
            input_ids[:len(combined_ids)] = combined_ids
            labels = torch.full((max_length,),-100,dtype=torch.long)
            labels[:len(combined_labels)] = combined_labels
            
            attention_mask=torch.zeros(max_length,dtype=torch.long)
            attention_mask[:len(combined_ids)]=1
            e["input_ids"] = input_ids
            e["labels"] = labels
            e["attention_mask"]=attention_mask
            return e
        self.dataset = dataset.map(process)
        self.dataset.set_format("torch")
    def __len__(self):
            return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]
        return data
        # return {"input_ids":data["input_ids"],
        #         "attention_mask":data["attention_mask"],
        #         "labels":data["labels"]}


def get_slot_movie_loader(batch_size=16,tokenizer=None,max_length=MAX_LENGTH,data_path=None,category="director",tag="train"):
    train_set = Slot_Movie_Dataset(tokenizer=tokenizer,split="train",data_path=data_path,category=category,tag=tag)    
    val_set = Slot_Movie_Dataset(tokenizer=tokenizer,split="test",data_path=data_path,category=category)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=8)
    return train_loader,val_loader