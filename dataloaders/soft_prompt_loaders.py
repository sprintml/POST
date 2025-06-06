import os
import random
import torch
from dataloaders.sst_loader import SST2_Dataset
from dataloaders.tweet_eval_loader import TweetEval_sentiment_Dataset
from torch.utils.data import DataLoader

class soft_prompt_SST2_Dataset(SST2_Dataset):
    def __init__(self,tokenizer,split,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,soft_prompt_path=None):
        super(soft_prompt_SST2_Dataset,self).__init__(tokenizer=tokenizer,
                                                 split=split,
                                                 text_infilling=text_infilling,
                                                 forpruning=forpruning,
                                                 cache_dir=cache_dir,
                                                 local_dir=local_dir)

        soft_prompt_names = os.listdir(soft_prompt_path)  
        self.soft_prompt_list = []
        for name in soft_prompt_names:
            file = os.path.join(soft_prompt_path,name)
            soft_prompt_dict = torch.load(file)
            soft_prompt = list(soft_prompt_dict.values())[0]
            self.soft_prompt_list.append(soft_prompt)

        self.soft_prompt_num = len(self.soft_prompt_list)

    def __getitem__(self,idx):
        data = self.dataset[idx]
        #return torch.IntTensor(data['input_ids']),torch.IntTensor(data["attention_mask"]),data['labels']
        # random pick one soft_prompt
        prompt_idx = random.randrange(self.soft_prompt_num)
        soft_prompt = self.soft_prompt_list[prompt_idx]
        if self.text_infilling:
            if not self.forpruning:
                return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]), 'labels':torch.LongTensor(data['labels']),'mask_idx':data['mask_idx'],'label':data['label'],'source_prompt':soft_prompt}
            else:
                return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]), 'labels':torch.LongTensor(data['labels'])}

        else:
            return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]), 'labels':data['labels'],'source_prompt':soft_prompt}

class soft_prompt_TweetEval_sentiment_Dataset(TweetEval_sentiment_Dataset):
    def __init__(self,tokenizer,split,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,soft_prompt_path=None):
        super(soft_prompt_TweetEval_sentiment_Dataset,self).__init__(tokenizer=tokenizer,
                                                 split=split,
                                                 text_infilling=text_infilling,
                                                 forpruning=forpruning,
                                                 cache_dir=cache_dir,
                                                 local_dir=local_dir)

        soft_prompt_names = os.listdir(soft_prompt_path)  
        self.soft_prompt_list = []
        for name in soft_prompt_names:
            file = os.path.join(soft_prompt_path,name)
            soft_prompt_dict = torch.load(file)
            soft_prompt = list(soft_prompt_dict.values())[0]
            self.soft_prompt_list.append(soft_prompt)

        self.soft_prompt_num = len(self.soft_prompt_list)

    def __getitem__(self,idx):
        data = self.dataset[idx]
        #return torch.IntTensor(data['input_ids']),torch.IntTensor(data["attention_mask"]),data['labels']
        # random pick one soft_prompt
        prompt_idx = random.randrange(self.soft_prompt_num)
        soft_prompt = self.soft_prompt_list[prompt_idx]
        if self.text_infilling:
            if not self.forpruning:
                return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]), 'labels':torch.LongTensor(data['labels']),'mask_idx':data['mask_idx'],'label':data['label'],'source_prompt':soft_prompt}
            else:
                return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]), 'labels':torch.LongTensor(data['labels'])}

        else:
            return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]), 'labels':data['labels'],'source_prompt':soft_prompt}



def get_soft_prompt_sst2_loader(batch_size=16,tokenizer=None,split=None,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,soft_prompt_path=None):
    assert (soft_prompt_path is not None),"soft_prompt_path is empty"
    dataset = soft_prompt_SST2_Dataset(tokenizer,split,text_infilling,forpruning,cache_dir,local_dir,soft_prompt_path)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    return loader

def get_soft_prompt_tweeteval_sentiment_loader(batch_size=16,tokenizer=None,split=None,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,soft_prompt_path=None):
    assert (soft_prompt_path is not None),"soft_prompt_path is empty"
    dataset = soft_prompt_TweetEval_sentiment_Dataset(tokenizer,split,text_infilling,forpruning,cache_dir,local_dir,soft_prompt_path)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    return loader