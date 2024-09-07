from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os
load_dotenv()

token = os.getenv("HUGGINGFACE")
model_name = "ek826/Meta-Llama-3.1-405B-Instruct-6.0bpw-exl2"
custom_cache_dir = "./llama-405b"

model_config = hf_hub_download(repo_id=model_name, filename="config.json", cache_dir=custom_cache_dir, token=token)
tokenizer_config = hf_hub_download(repo_id=model_name, filename="tokenizer_config.json", cache_dir=custom_cache_dir, token=token)
generation_config = hf_hub_download(repo_id=model_name, filename="generation_config.json", cache_dir=custom_cache_dir, token=token)

special_tokens =  hf_hub_download(repo_id=model_name, filename="special_tokens_map.json", cache_dir=custom_cache_dir, token=token)
tokenizer = hf_hub_download(repo_id=model_name, filename="tokenizer.json", cache_dir=custom_cache_dir, token=token)
index = hf_hub_download(repo_id=model_name, filename="model.safetensors.index.json", cache_dir=custom_cache_dir, token=token)

patch = hf_hub_download(repo_id=model_name, filename="patch.diff", cache_dir=custom_cache_dir, token=token)

weights = []
for i in range(1, 39):
    weights.append(hf_hub_download(repo_id=model_name, filename=f"output-{i:05}-of-00038.safetensors", cache_dir=custom_cache_dir, token=token))
