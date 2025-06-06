import torch
import random
import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import concatenate_datasets

MAX_LENGTH = 128
Text_labels = {0:"negative",1:"positive"}
#Text_labels = {0:"bad",1:"good"}

class SST2_MC_Dataset(Dataset):
    def __init__(self,tokenizer,split,max_length=MAX_LENGTH,seed=None,prompt=True,prompt_length=0):
        if prompt:
            assert prompt_length>0
        self.prompot_length = prompt_length

        if split=="train":
            dataset = load_dataset('glue', 'sst2', split='train')
        elif split == "val":
            dataset = load_dataset('glue', 'sst2', split='validation')
        elif split == "test":
            dataset = load_dataset('glue', 'sst2', split='test')


        def process_infilling(example):
            reverse = random.random()>0.5
            if reverse:
                A = "positive"
                B = "negative"
                example['label'] = 1-example['label']
            else:
                A="negative"
                B="positive"
            example["sentence"] = example["sentence"]+f". Determine if this review is positive or negative, choose between A and B where A represents {A}, B represents {B}. This review is <mask>."
            return example
        dataset = dataset.map(process_infilling)
                
        tokenizer.truncation_side = "left"
        self.dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=max_length), batched=False)

        def get_mask_idx(example):
            idx=(torch.Tensor(example['input_ids'])==tokenizer.mask_token_id).nonzero().item()
            example['mask_idx'] = idx
            return example
        self.dataset = self.dataset.map(get_mask_idx)
        if seed:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.text_label=[" A"," B"]
        self.label_idx = [tokenizer.encode(text_label)[1] for text_label in self.text_label]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]

        return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),'mask_idx':data['mask_idx']+self.prompot_length,'label':data['label']}


class IMDB_MC_Dataset(Dataset):
    def __init__(self,tokenizer,split,max_length=MAX_LENGTH,seed=None,prompt=True,prompt_length=0):
        if prompt:
            assert prompt_length>0
        self.prompot_length = prompt_length
        dataset = load_dataset("imdb")

        if split=="train":
            dataset = dataset['train']
        elif split == "val":
            dataset = dataset['val']


        def process_infilling(example):
            reverse = random.random()>0.5
            if reverse:
                A = "positive"
                B = "negative"
                example['label'] = 1-example['label']
            else:
                A="negative"
                B="positive"
            example["sentence"] = example["text"]+f".  Determine if this review is positive or negative, choose between A and B where A represents {A}, B represents {B}. It is <mask>."
            return example
        dataset = dataset.map(process_infilling)
                
        tokenizer.truncation_side = "left"
        self.dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=max_length), batched=False)

        def get_mask_idx(example):
            idx=(torch.Tensor(example['input_ids'])==tokenizer.mask_token_id).nonzero().item()
            example['mask_idx'] = idx
            return example
        self.dataset = self.dataset.map(get_mask_idx)
        if seed:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.text_label=[" A"," B"]
        self.label_idx = [tokenizer.encode(text_label)[1] for text_label in self.text_label]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]

        return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),'mask_idx':data['mask_idx']+self.prompot_length,'label':data['label']}


class Tweeteval_sentiment_MC_Dataset(Dataset):
    def __init__(self,tokenizer,split,max_length=MAX_LENGTH,seed=None,prompt=True,prompt_length=0):
        if prompt:
            assert prompt_length>0
        self.prompot_length = prompt_length
        dataset = load_dataset("tweet_eval", "sentiment")

        if split=="train":
            dataset = dataset['train']
        elif split == "val":
            dataset = dataset['val']
        elif split =="test":
            dataset = dataset['test']


        def process_infilling(example):
            random_num = random.random()
            if random_num<1/6:
                A = "negative"
                B = "neutral"
                C = "positive"
            elif random_num<2/6:
                A =  "negative"
                B = "positive"
                C = "neutral"
                label_list = [0,2,1]
                example['label'] = label_list[example['label']]
            elif random_num<3/6:
                A =  "neutral"
                B = "positive"
                C = "negative"
                label_list = [2,0,1]
                example['label'] = label_list[example['label']]
            elif random_num<4/6:
                A =  "neutral"
                B = "negative"
                C = "positive"
                label_list = [1,0,2]
                example['label'] = label_list[example['label']]
            elif random_num<5/6:
                A =  "positive"
                B = "negative"
                C = "neutral"
                label_list = [1,2,0]
                example['label'] = label_list[example['label']]
            else:
                A =  "positive"
                B = "neutral"
                C = "negative"
                label_list = [2,1,0]
                example['label'] = label_list[example['label']]
            
            example["sentence"] = example["text"]+f".  Determine if this review is positive, negative or neutral, choose between A, B and C where A represents {A}, B represents {B} c represent {C}. It is <mask>."
            return example
        dataset = dataset.map(process_infilling)
                
        tokenizer.truncation_side = "left"
        self.dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=max_length), batched=False)

        def get_mask_idx(example):
            idx=(torch.Tensor(example['input_ids'])==tokenizer.mask_token_id).nonzero().item()
            example['mask_idx'] = idx
            return example
        self.dataset = self.dataset.map(get_mask_idx)
        if seed:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.text_label=[" A"," B"," C"]
        self.label_idx = [tokenizer.encode(text_label)[1] for text_label in self.text_label]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]

        return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),'mask_idx':data['mask_idx']+self.prompot_length,'label':data['label']}


class QNLI_MC_Dataset(Dataset):
    def __init__(self,tokenizer,split,max_length=MAX_LENGTH,seed=None,prompt=True,prompt_length=0):
        if prompt:
            assert prompt_length>0
        self.prompot_length = prompt_length

        if split=="train":
            dataset = load_dataset('glue', 'qnli', split='train')
        elif split == "val":
            dataset = load_dataset('glue', 'qnli', split='validation')
        elif split == "test":
            dataset = load_dataset('glue', 'qnli', split='test')


        def process_infilling(example):
            reverse = random.random()>0.5
            if reverse:
                A = "no"
                B = "yes"
                example['label'] = 1-example['label']
            else:
                A="yes"
                B="no"
            example["sentence"] = "Question: "+example["question"]+" Sentence: "+example["sentence"]+f" Determine if the sentence contains answer of the question, choose between A and B where A represents {A}, B represents {B}. It is <mask>."
            return example
        dataset = dataset.map(process_infilling)
                
        tokenizer.truncation_side = "left"
        self.dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=max_length), batched=False)

        def get_mask_idx(example):
            idx=(torch.Tensor(example['input_ids'])==tokenizer.mask_token_id).nonzero().item()
            example['mask_idx'] = idx
            return example
        self.dataset = self.dataset.map(get_mask_idx)
        if seed:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.text_label=[" A"," B"]
        self.label_idx = [tokenizer.encode(text_label)[1] for text_label in self.text_label]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]

        return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),'mask_idx':data['mask_idx']+self.prompot_length,'label':data['label']}
    

class MNLI_MC_Dataset(Dataset):
    def __init__(self,tokenizer,split,max_length=MAX_LENGTH,seed=None,prompt=True,prompt_length=0):
        if prompt:
            assert prompt_length>0
        self.prompot_length = prompt_length

        if split=="train":
            dataset = load_dataset('glue', 'mnli', split="train")
        elif split == "val":
            dataset1 = load_dataset("glue", "mnli",split="validation_matched")
            dataset2 = load_dataset("glue", "mnli",split="validation_mismatched")
            dataset = concatenate_datasets([dataset1,dataset2])
        elif split =="test":
            dataset = load_dataset('glue', 'mnli', split="test")


        def process_infilling(example):
            random_num = random.random()
            if random_num<1/6:
                A = "yes"
                B = "maybe"
                C = "no"
            elif random_num<2/6:
                A =  "yes"
                B = "no"
                C = "maybe"
                label_list = [0,2,1]
                example['label'] = label_list[example['label']]
            elif random_num<3/6:
                A =  "maybe"
                B = "no"
                C = "yes"
                label_list = [2,0,1]
                example['label'] = label_list[example['label']]
            elif random_num<4/6:
                A =  "maybe"
                B = "yes"
                C = "no"
                label_list = [1,0,2]
                example['label'] = label_list[example['label']]
            elif random_num<5/6:
                A =  "no"
                B = "yes"
                C = "maybe"
                label_list = [1,2,0]
                example['label'] = label_list[example['label']]
            else:
                A =  "no"
                B = "maybe"
                C = "yes"
                label_list = [2,1,0]
                example['label'] = label_list[example['label']]
            
            example["sentence"]="Premise::" +example['premise'] + " Hypothesis: " +example["hypothesis"]+f" Determine if the premise entails the hypothesis, choose between A, B and C where A represents {A}, B represents {B}, C represents {C}. It is <mask>  "
            return example
        dataset = dataset.map(process_infilling)
                
        tokenizer.truncation_side = "left"
        self.dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=max_length), batched=False)

        def get_mask_idx(example):
            idx=(torch.Tensor(example['input_ids'])==tokenizer.mask_token_id).nonzero().item()
            example['mask_idx'] = idx
            return example
        self.dataset = self.dataset.map(get_mask_idx)
        if seed:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.text_label=[" A"," B"," C"]
        self.label_idx = [tokenizer.encode(text_label)[1] for text_label in self.text_label]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]

        return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),'mask_idx':data['mask_idx']+self.prompot_length,'label':data['label']}

class SNLI_MC_Dataset(Dataset):
    def __init__(self,tokenizer,split,max_length=MAX_LENGTH,seed=None,prompt=True,prompt_length=0):
        if prompt:
            assert prompt_length>0
        self.prompot_length = prompt_length

        dataset = load_dataset('snli', split=split)
        dataset = dataset.select(
            (
                i for i,data in enumerate(dataset) 
                if data['label'] !=-1)
            )
        tokenizer.truncation_side = "left"

        def process_infilling(example):
            random_num = random.random()
            if random_num<1/6:
                A = "yes"
                B = "maybe"
                C = "no"
            elif random_num<2/6:
                A =  "yes"
                B = "no"
                C = "maybe"
                label_list = [0,2,1]
                example['label'] = label_list[example['label']]
            elif random_num<3/6:
                A =  "maybe"
                B = "no"
                C = "yes"
                label_list = [2,0,1]
                example['label'] = label_list[example['label']]
            elif random_num<4/6:
                A =  "maybe"
                B = "yes"
                C = "no"
                label_list = [1,0,2]
                example['label'] = label_list[example['label']]
            elif random_num<5/6:
                A =  "no"
                B = "yes"
                C = "maybe"
                label_list = [1,2,0]
                example['label'] = label_list[example['label']]
            else:
                A =  "no"
                B = "maybe"
                C = "yes"
                label_list = [2,1,0]
                example['label'] = label_list[example['label']]
            
            example["sentence"]="Premise::" +example['premise'] + " Hypothesis: " +example["hypothesis"]+f" Determine if the premise entails the hypothesis, choose between A, B and C where A represents {A}, B represents {B}, C represents {C}. It is <mask>  "
            return example
        dataset = dataset.map(process_infilling)
                
        tokenizer.truncation_side = "left"
        self.dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=max_length), batched=False)

        def get_mask_idx(example):
            idx=(torch.Tensor(example['input_ids'])==tokenizer.mask_token_id).nonzero().item()
            example['mask_idx'] = idx
            return example
        self.dataset = self.dataset.map(get_mask_idx)
        if seed:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.text_label=[" A"," B"," C"]
        self.label_idx = [tokenizer.encode(text_label)[1] for text_label in self.text_label]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        data = self.dataset[idx]

        return {'input_ids':torch.IntTensor(data['input_ids']), 'attention_mask':torch.IntTensor(data["attention_mask"]),'mask_idx':data['mask_idx']+self.prompot_length,'label':data['label']}

