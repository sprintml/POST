# Efficient and Privacy-Preserving Soft Prompt Transfer for LLMs

*Keywords:* prompt transfer, soft prompt, privacy, distillation, confidentiality

*TL;DR:* We propose a method on how to transfer soft prompts tuned on a distilled small model to a larger model using public data.

*Abstract:*
Prompting has become a dominant paradigm for adapting large language models (LLMs). While discrete (textual) prompts are widely used for their interpretability, soft (parameter) prompts have recently gained traction in APIs. This is because they can encode information from more training samples while minimizing the user's token usage, leaving more space in the context window for task-specific input. However, soft prompts are tightly coupled to the LLM they are tuned on, limiting their generalization to other LLMs. This constraint is particularly problematic for efficiency and privacy: (1) tuning prompts on each LLM incurs high computational costs, especially as LLMs continue to grow in size. Additionally, (2) when the LLM is hosted externally, soft prompt tuning often requires sharing private data with the LLM provider. For instance, this is the case with the NVIDIA NeMo API. To address these issues, we propose POST (Privacy Of Soft prompt Transfer), a framework that enables private tuning of soft prompts on a small model and subsequently transfers these prompts to a larger LLM. POST uses knowledge distillation to derive a small model directly from the large LLM to improve prompt transferability, tunes the soft prompt locally---optionally with differential privacy guarantees---and transfers it back to the larger LLM using a small public dataset. Our experiments show that POST reduces computational costs, preserves privacy, and effectively transfers high-utility soft prompts..
## requirements

`pip install -r requirements.txt`

## usage
1. prepare dataset    
use `disll/process_data.py` and `disll/token_counts.py` to generate dataset used for knowledge distillation  
use `disll/extract_distill_model.py` to get the initialization of source model   
use `distill/train.py` to get the dsitlled source model 
2. train soft prompt on source model  
run `scripts/train_prompt.py` to train soft prompt without DP  
run `scripts/train_DP_prompt.py` to train soft prompt with DP  
3. prompt transfer
run `scripts/prompt_transfer.py` to transfer prompt need to specify the public dataset
4. perform LiRA
run `scripts/train_lira_models.py` to train LiRA models and perform inference, need to specify number of reference models

