import torch
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

Text_labels = {0:"yes",1:"no"}
MAX_LENGTH=128

class QNLI_Dataset(Dataset):
    def __init__(self,tokenizer,split,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,max_length=MAX_LENGTH,seed=None):

        if local_dir:
            dataset = datasets.load_from_disk(local_dir)[split]
        elif cache_dir:
            dataset = load_dataset('glue', 'qnli', split=split,cache_dir=cache_dir,download_mode="reuse_cache_if_exists")
        else:
            dataset = load_dataset('glue', 'qnli', split=split)
        # use right truncation since questions are normally shorter
        tokenizer.truncation_side = "right"
        self.text_infilling = text_infilling
        if text_infilling:
            self.dataset = dataset.map(lambda e: tokenizer(e['question']+"? <mask>,"+e['sentence'], truncation=True, padding='max_length', max_length=max_length), batched=False)
            def create_idx(e):
                idx=(torch.Tensor(e['input_ids'])==tokenizer.mask_token_id).nonzero().item()
                e['mask_idx'] = idx
                return e
            self.dataset = self.dataset.map(create_idx)
        else:
            self.dataset = dataset.map(lambda e: tokenizer(e['question']+tokenizer.sep_token+e['sentence'], truncation=True, padding='max_length', max_length=max_length), batched=False)
        self.forpruning = forpruning
        if seed:
            self.dataset = self.dataset.shuffle(seed=seed)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]
        if self.text_infilling:
            return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),"mask_idx":data["mask_idx"],"label":data['label']}
        else:
            return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),"mask_idx":0,"label":data['label']}

def get_qnli_loader(batch_size=16,tokenizer=None,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,max_length=MAX_LENGTH):
    train_dataset = QNLI_Dataset(tokenizer,"train",text_infilling,forpruning,cache_dir,local_dir,max_length=max_length)
    val_dataset = QNLI_Dataset(tokenizer,"validation",text_infilling,forpruning,cache_dir,local_dir,max_length=max_length)
    #test_dataset = SST2_Dataset(tokenizer,"test") # test dataset has -1 as labels
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=8)
    #test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,val_loader
