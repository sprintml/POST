import torch
import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset

MAX_LENGTH = 256
Text_labels = {0:"negative",1:"moderate",2:"positive"}
#Text_labels = {0:"bad",1:"neutral",2:"good"}

class TweetEval_sentiment_Dataset(Dataset):
    def __init__(self,tokenizer,split,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,max_length=MAX_LENGTH,seed=None,task="mlm"):
        if local_dir:
            dataset = datasets.load_from_disk(local_dir)
        elif cache_dir:
            dataset = load_dataset("tweet_eval","sentiment",cache_dir=cache_dir,download_mode="reuse_cache_if_exists")
        else:
            dataset = load_dataset("tweet_eval", "sentiment")
        if split=="train":
            dataset = dataset['train']
        elif split=="val":
            dataset = dataset['validation']
        elif split=="test":
            dataset = dataset["test"]

        self.text_infilling = text_infilling
        dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        if text_infilling:
            if task=="mlm":
                def process_infilling(example):
                    example["sentence"] = example["text"]+"It was <mask>."
                    example['text_labels'] = Text_labels[example['label']]
                    return example
            elif task=="clm":
                def process_infilling(example):
                    example["sentence"] = example["text"]+"It was"
                    example['text_labels'] = Text_labels[example['label']]
                    return example
            dataset = dataset.map(process_infilling)
        else:
            def process(example):
                example["sentence"] = example["text"]
                return example
            dataset = dataset.map(process)
        # set tokenizer truncate from left
        tokenizer.truncation_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        self.dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=max_length), batched=False)
        if text_infilling:
            if task=="mlm":
                def create_labels(example):
                    example['labels'] = [-100] * len(example['input_ids'])
                    idx=(torch.Tensor(example['input_ids'])==tokenizer.mask_token_id).nonzero().item()
                    example['labels'][idx] = tokenizer.encode(example['text_labels'])[1] # only take 
                    example['mask_idx'] = idx
                    return example
            elif task=="clm":
                def create_labels(example):
                    example['labels'] = [-100] * len(example['input_ids'])
                    zero_idx=(torch.Tensor(example['input_ids'])==tokenizer.pad_token_id).nonzero()
                    if zero_idx.shape[0]==0: # length larger than max length
                        idx = len(example['input_ids'])-1 # take the last index
                    else:
                        idx = zero_idx[0].item()-1
                    example['labels'][idx] = tokenizer.encode(example['text_labels'])[0]  
                    example['mask_idx'] = idx # last meaningful token
                    return example
            self.dataset = self.dataset.map(create_labels)
        self.forpruning = forpruning
        self.task = task
        if seed:
            self.dataset=self.dataset.shuffle(seed=seed)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]
        #return torch.IntTensor(data['input_ids']),torch.IntTensor(data["attention_mask"]),data['labels']
        if self.text_infilling:
            if not self.forpruning:
                return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),'mask_idx':data['mask_idx'],'label':data['label']}
            else: # for pruning using textPruner
                return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"])}

        else:
            return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),"mask_idx":0,"label":data['label']}

def get_tweeteval_sentiment_loader(batch_size=16,tokenizer=None,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,max_length=MAX_LENGTH,seed=None,task="mlm"):
    train_dataset = TweetEval_sentiment_Dataset(tokenizer,"train",text_infilling,forpruning,cache_dir,local_dir,max_length=max_length,seed=seed,task=task)
    val_dataset = TweetEval_sentiment_Dataset(tokenizer,"val",text_infilling,forpruning,cache_dir,local_dir,max_length=max_length,seed=seed,task=task)
    #test_dataset = SST2_Dataset(tokenizer,"test") # test dataset has -1 as labels
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=8)
    #test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,val_loader