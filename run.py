import sys
import time

sys.path.append("mixtral-offloading")

import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer

from transformers import TextStreamer

from src.build_model import OffloadConfig, QuantConfig, build_model

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"

config = AutoConfig.from_pretrained(quantized_model_name)

device = torch.device("cuda:0")

##### Change this to 5 if you have only 12 GB of GPU VRAM #####
offload_per_layer = 4
# offload_per_layer = 5
###############################################################

num_experts = config.num_local_experts

offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
)


attn_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)
attn_config["scale_quant_params"]["group_size"] = 256


ffn_config = BaseQuantizeConfig(
    nbits=2,
    group_size=16,
    quant_zero=True,
    quant_scale=True,
)
quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)


model, expert_cache = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=state_path,
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
past_key_values = None
sequence = None

seq_len = 0
# while True:
# print("User: ", end="")
# user_input = input()
# print("\n")

user_input = "Tell me about India in 10 words."
user_entry = dict(role="user", content=user_input)
input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

if past_key_values is None:
    attention_mask = torch.ones_like(input_ids)
else:
    seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
    attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

start_time = time.time()
print("Mixtral: ", end="")
result = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    streamer=streamer,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_hidden_states=True,
)
print("\n")
time_taken = time.time() - start_time

sequence = result["sequences"]
past_key_values = result["past_key_values"]

# get the total input and output sequence length
seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
print("Total sequence length: ", seq_len)

# get tok/sec
tok_per_sec = seq_len / time_taken
print("Tokens per second: ", tok_per_sec)

# print #hits and #misses
layers2hits = {}
layers2misses = {}
expert_hits = {}
for layer_idx, eviction_group_info in expert_cache.group_infos.items():
    hits = eviction_group_info.hits
    misses = eviction_group_info.misses
    layers2hits[layer_idx] = hits
    layers2misses[layer_idx] = misses
    print(f"Layer {layer_idx}: Hits: {hits}, Misses: {misses}")
    expert_hits[layer_idx] = eviction_group_info.expert_hits

print("Total hits: ", sum(layers2hits.values()))
print("Total misses: ", sum(layers2misses.values()))
print(expert_hits)

# # plot a heat map of per layer expert hits
# import seaborn as sns
# import matplotlib.pyplot as plt

# # expert_hits[i][j] is the number of times expert j was hit in layer i
# expert_hits = [expert_hits[i].reshape(-1, 1) for i in range(len(expert_hits))]

# sns.heatmap(expert_hits, annot=True, fmt="d")
# plt.xlabel("Experts")
# plt.savefig("expert_hits.png")
