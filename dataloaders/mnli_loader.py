import torch
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
Text_labels = {0:"yes",1:"moderate",2:"no"}
MAX_LENGTH=128

class MNLI_Dataset(Dataset):
    def __init__(self,tokenizer,split,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,max_length=MAX_LENGTH,seed=None):
        if split=="val":
            dataset1 = load_dataset("glue", "mnli",split="validation_matched")
            dataset2 = load_dataset("glue", "mnli",split="validation_mismatched")
            dataset = concatenate_datasets([dataset1,dataset2])
        else:
            if local_dir:
                dataset = datasets.load_from_disk(local_dir)[split]
            elif cache_dir:
                dataset = load_dataset('glue', 'mnli', split=split,cache_dir=cache_dir,download_mode="reuse_cache_if_exists")
            else:
                dataset = load_dataset('glue', 'mnli', split=split)

        tokenizer.truncation_side = "left"
        self.text_infilling = text_infilling
        if text_infilling:
            self.dataset = dataset.map(lambda e: tokenizer(e['premise']+"? <mask>,"+e['hypothesis'], truncation=True, padding='max_length', max_length=max_length), batched=False)
            def create_idx(e):
                idx=(torch.Tensor(e['input_ids'])==tokenizer.mask_token_id).nonzero().item()
                e['mask_idx'] = idx
                return e
            self.dataset = self.dataset.map(create_idx)
        else:
            self.dataset = dataset.map(lambda e: tokenizer(e['premise']+tokenizer.sep_token+e['hypothesis'], truncation=True, padding='max_length', max_length=max_length), batched=False)
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


def get_mnli_loader(batch_size=16,tokenizer=None,text_infilling=False,forpruning=False,cache_dir=None,local_dir=None,max_length=MAX_LENGTH):
    train_dataset = MNLI_Dataset(tokenizer,"train",text_infilling,forpruning,cache_dir,local_dir,max_length=max_length)
    val_match_dataset = MNLI_Dataset(tokenizer,"validation_matched",text_infilling,forpruning,cache_dir,local_dir,max_length=max_length)
    val_mismatch_dataset = MNLI_Dataset(tokenizer,"validation_mismatched",text_infilling,forpruning,cache_dir,local_dir,max_length=max_length)
    #test_dataset = SST2_Dataset(tokenizer,"test") # test dataset has -1 as labels
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=8)
    val_match_loader = DataLoader(val_match_dataset,batch_size=batch_size,shuffle=False,num_workers=8)
    val_mismatch_loader = DataLoader(val_mismatch_dataset,batch_size=batch_size,shuffle=False,num_workers=8)
    #test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,val_match_loader, val_mismatch_loader
