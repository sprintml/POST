import os
import random
import torch
import random
from torch.utils.data import Dataset
from dataloaders.sst_loader import SST2_Dataset
from dataloaders.tweet_eval_loader import TweetEval_sentiment_Dataset
from dataloaders.imdb_loader import IMDB_Dataset
from dataloaders.fpb_loader import FPB_Dataset, get_data_and_split
from dataloaders.qnli_loader import QNLI_Dataset
from dataloaders.mnli_loader import MNLI_Dataset
from dataloaders.snli_loader import SNLI_Dataset
from torch.utils.data import DataLoader
from bisect import bisect_right
MAX_LENGTH=256

def get_dataset(dataset_name,tokenizer,split,text_infilling=True,forpruning=False,cache_dir=None,local_dir=None,path_dict=None,max_length=MAX_LENGTH):
    if dataset_name=="sst2":
        dataset = SST2_Dataset(tokenizer=tokenizer,split=split,text_infilling=text_infilling,forpruning=forpruning,cache_dir=cache_dir,local_dir=local_dir,max_length=max_length)
    elif dataset_name =="tweeteval_sentiment":
        dataset = TweetEval_sentiment_Dataset(tokenizer=tokenizer,split=split,text_infilling=text_infilling,forpruning=forpruning,cache_dir=cache_dir,local_dir=local_dir,max_length=max_length)
    elif dataset_name == "imdb":
        dataset = IMDB_Dataset(tokenizer=tokenizer,split=split,text_infilling=text_infilling,forpruning=forpruning,cache_dir=cache_dir,local_dir=local_dir,max_length=max_length)
    elif dataset_name == "fpb":
        root = path_dict["fpb"]['root']
        text_train,text_val,labels_train,labels_val = get_data_and_split(root)
        if split=="train":
            dataset = FPB_Dataset(text_train,labels_train,tokenizer=tokenizer,text_infilling=text_infilling,max_length=max_length)
        elif split=="val":
            dataset = FPB_Dataset(text_val,labels_val,tokenizer=tokenizer,text_infilling=text_infilling,max_length=max_length)
    elif dataset_name=="qnli":
        if split == "train":
            dataset = QNLI_Dataset(tokenizer=tokenizer,split="train",text_infilling=text_infilling,forpruning=forpruning,cache_dir=cache_dir,local_dir=local_dir,max_length=max_length)
        elif split=="val":
            dataset = QNLI_Dataset(tokenizer=tokenizer,split="validation",text_infilling=text_infilling,forpruning=forpruning,cache_dir=cache_dir,local_dir=local_dir,max_length=max_length)
    elif dataset_name =="mnli":
        dataset = MNLI_Dataset(tokenizer=tokenizer,split=split,text_infilling=text_infilling,forpruning=forpruning,cache_dir=cache_dir,local_dir=local_dir,max_length=max_length)
    elif dataset_name=="snli":
        if split == "train":
            dataset = SNLI_Dataset(tokenizer=tokenizer,split="train",text_infilling=text_infilling,forpruning=forpruning,cache_dir=cache_dir,local_dir=local_dir,max_length=max_length)
        elif split=="val":
            dataset = SNLI_Dataset(tokenizer=tokenizer,split="validation",text_infilling=text_infilling,forpruning=forpruning,cache_dir=cache_dir,local_dir=local_dir,max_length=max_length)
    return dataset


class Multi_Prompt_Dataset(Dataset):
    def __init__(self,dataset_list,tokenizer,split,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,path_dict=None,balance=True,prompt_length=100):
        print(f"prompt length: {prompt_length}")
        self.prompt_length = prompt_length
        self.dataset_dict ={}
        self.dataset_length = {}
        self.soft_prompt_dict={}
        self.soft_prompt_num = {}
        for dataset_name in dataset_list:
            # get dataset
            dataset =  get_dataset(dataset_name=dataset_name,tokenizer=tokenizer,split=split,text_infilling=text_infilling,
                                   forpruning=forpruning,cache_dir=cache_dir,local_dir=local_dir,path_dict=path_dict)
            self.dataset_dict[dataset_name] = dataset
            self.dataset_length[dataset_name] = len(dataset)
            # get soft prompt dataset
            soft_prompt_path = path_dict[dataset_name][split]
            soft_prompt_names = os.listdir(soft_prompt_path)  
            soft_prompt_list = []
            for name in soft_prompt_names:
                file = os.path.join(soft_prompt_path,name)
                soft_prompt_dict = torch.load(file)
                soft_prompt = list(soft_prompt_dict.values())[0]
                soft_prompt_list.append(soft_prompt)
            self.soft_prompt_dict[dataset_name]=soft_prompt_list
            self.soft_prompt_num[dataset_name] = len(soft_prompt_list)
        # balance dataset num in trainset
        if split=="train" and balance==True:
            num = min(list(self.dataset_length.values()))
            random.seed(42)
            for dataset_name in self.dataset_length:
                ori_num = self.dataset_length[dataset_name]
                sample_list=random.sample(range(0, ori_num), num)
                self.dataset_dict[dataset_name] = torch.utils.data.Subset(self.dataset_dict[dataset_name], sample_list)
                self.dataset_length[dataset_name] = num
        self.create_search_list()
        self.text_infilling = text_infilling
        self.forpruning = forpruning

    def __len__(self):
        num=0
        for dataset in self.dataset_dict:
            num+=self.dataset_length[dataset]
        return num

    def __getitem__(self, idx):
        dataset_name,local_idx = self.index2dataset(idx)
        data = self.dataset_dict[dataset_name][local_idx]
        prompt_idx = random.randrange(self.soft_prompt_num[dataset_name])
        soft_prompt = self.soft_prompt_dict[dataset_name][prompt_idx]
        
        if self.text_infilling:
            if not self.forpruning:
                return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),'mask_idx':data['mask_idx']+self.prompt_length,'label':data['label'],'source_prompt':soft_prompt,"dataset":dataset_name}
            else:
                return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),"dataset":dataset_name}

        else:
            return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),'mask_idx':data['mask_idx']+self.prompt_length,'label':data['label'],'source_prompt':soft_prompt,"dataset":dataset_name}
    def create_search_list(self):
        self.search_list=[0]
        self.order2dataset ={}
        n=0
        i=0
        for dataset_name in self.dataset_length:
            self.order2dataset[i] = dataset_name
            n+=self.dataset_length[dataset_name]
            self.search_list.append(n)
            i+=1

    
    def index2dataset(self,index):

        order = bisect_right(self.search_list,index)-1
        dataset_name = self.order2dataset[order]
        local_index = index-self.search_list[order]

        return dataset_name, local_index
        
    
def get_multi_prompt_loader(batch_size=16,dataset_list=None,tokenizer=None,split=None,text_infilling=True,forpruning=False,cache_dir=None,local_dir=None,path_dict=None,balance=True):
    dataset = Multi_Prompt_Dataset(dataset_list=dataset_list,tokenizer=tokenizer,split=split,text_infilling=text_infilling,forpruning=forpruning,cache_dir=cache_dir,local_dir=local_dir,path_dict=path_dict,balance=balance)
    if split=="train":
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    else:
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=8)
    return loader

        
