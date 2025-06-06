import os
import torch
import torch.nn as nn
from pathlib import Path
from utils.utils import process_config
from transformers import RobertaForSequenceClassification,RobertaConfig,RobertaForMaskedLM

class RobertaMixin:
    ## todo add from_models so that don't need to save to continue

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        soft_prompt_path: str = None,
                        n_tokens: int = None,
                        initialize_from_vocab: bool = True,
                        random_range: float = 0.5,
                        hard_prompt_ids: list= None,
                        **kwargs,):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        else:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
                hard_prompt_ids=hard_prompt_ids,
            )
        model.from_adaptor=False
        return model
    
    @classmethod
    def from_local(cls,
                   config_path,
                   state_path,
                   soft_prompt_path: str = None,
                   n_tokens: int = None,
                   initialize_from_vocab: bool = True,
                   random_range: float = 0.5,
                    hard_prompt_ids: list= None,
                   **kwargs,):
        config = RobertaConfig.from_json_file(config_path)
        model = cls(config)
        # need to manually set target_ffn_size
        # load state_dict
        model.load_state_dict(torch.load(state_path))
        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False
        
        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        else:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
                hard_prompt_ids=hard_prompt_ids,
            )
        model.from_adaptor=False
        return model


    @classmethod
    def from_pruned(cls,
                    path,
                    target_ffn_size,
                    soft_prompt_path: str = None,
                    n_tokens: int = None,
                    initialize_from_vocab: bool = True,
                    random_range: float = 0.5,
                    hard_prompt_ids: list= None,
                    **kwargs,):
        config_path = os.path.join(path,"config.json")
        config_dict = process_config(config_path)
        config = RobertaConfig.from_dict(config_dict)
        model = cls(config)
        # need to manually set target_ffn_size
        for layer in model.roberta.encoder.layer:
            n_in = layer.intermediate.dense.in_features
            layer.intermediate.dense = nn.Linear(in_features=n_in,out_features= target_ffn_size)
            n_out = layer.output.dense.out_features
            layer.output.dense = nn.Linear(in_features= target_ffn_size, out_features=n_out)
        # load state_dict
        state_path = os.path.join(path,"pytorch_model.bin")
        model.load_state_dict(torch.load(state_path))
        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False
        
        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        else:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
                hard_prompt_ids=hard_prompt_ids,
            )
        model.from_adaptor=False
        return model

    @classmethod
    def from_adaptor(cls,
                        pretrained_model_name_or_path,
                        embed_dim: int =768,
                        num_heads: int=12,
                        n_tokens: int=20,
                        adaptor_path:str=None,
                        **kwargs,):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False
        if adaptor_path:
            model.set_adaptor(adaptor_path,n_tokens)
        else:
            print("Initializing soft prompt from adaptor...")
            model.initialize_adaptors(embed_dim = embed_dim,
                            num_heads = num_heads,
                            n_tokens =n_tokens
                            )

        return model
    
    def set_adaptor(self,adaptor_path:str,n_tokens):
        self.from_adaptor = True
        self.adaptor = torch.load(
            adaptor_path, map_location=torch.device("cpu")
        )
        self.n_tokens = n_tokens
    
    
    def initialize_adaptors(
        self,
        embed_dim,
        num_heads,
        n_tokens,


    ) -> None:
        self.adaptor = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        self.from_adaptor = True
        self.n_tokens = n_tokens

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path

        """
        soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device("cpu")
        )
        if isinstance(soft_prompt,dict):
            soft_prompt = list(soft_prompt.values())[0]
            self.n_tokens = soft_prompt.size()[0]
            self.embed_dim = self.roberta.embeddings.word_embeddings.weight.shape[1]
            self.soft_prompt = nn.Embedding(self.n_tokens, self.embed_dim)
            self.soft_prompt.weight = nn.parameter.Parameter(soft_prompt)
        else:
            self.soft_prompt = soft_prompt    
            self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
        self,
        n_tokens: int = 20,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        hard_prompt_ids: list = None,
    ) -> None:
        if hard_prompt_ids:
            print("init from hard prompts")
            self.n_tokens= len(hard_prompt_ids)
            self.embed_dim = self.roberta.embeddings.word_embeddings.weight.shape[1]
            init_prompt_value = self.roberta.embeddings.word_embeddings.weight[hard_prompt_ids,:].clone().detach()
        else:
            self.n_tokens = n_tokens
            self.embed_dim = self.roberta.embeddings.word_embeddings.weight.shape[1]
            if initialize_from_vocab:
                print("init from vocab")
                init_prompt_value = self.roberta.embeddings.word_embeddings.weight[:n_tokens].clone().detach()
            else:
                print("init from random")
                init_prompt_value = torch.rand(self.n_tokens, self.embed_dim) * 2 * random_range - random_range
        self.soft_prompt = nn.Embedding(self.n_tokens, self.embed_dim)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids,source_prompt) -> torch.Tensor:
        inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)
        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if self.from_adaptor:
            learned_embeds = self.adaptor(source_prompt,source_prompt,source_prompt,need_weights=False)[0]
            # repeat if only one is given
            if len(learned_embeds.shape)<len(inputs_embeds.shape):
                learned_embeds = learned_embeds.repeat(inputs_embeds.size(0), 1, 1)
        else:
            learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
        # [batch_size, n_tokens, n_embd]
     
        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        os.makedirs(path,exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        print(f"Saved soft prompt: {os.path.join(path, filename)}")

    def save_adaptor(self,path:str,filename:str="adaptor.model"):
        os.makedirs(path,exist_ok=True)
        torch.save(self.adaptor, os.path.join(path, filename))
        print(f"Saved adaptor: {os.path.join(path, filename)}")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        source_prompt=None,
    ):
        if input_ids is not None:
            if self.from_adaptor:
                inputs_embeds = self._cat_learned_embedding_to_input(input_ids,source_prompt)
            else:
                inputs_embeds = self._cat_learned_embedding_to_input(input_ids,None).to(
                    self.device
                )

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class RobertaPTSeqCls(RobertaMixin, RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

class RobertaPTMLM(RobertaMixin,RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)