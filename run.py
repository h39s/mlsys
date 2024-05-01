import sys
import time
import json
import numpy as np

np.random.seed(42)

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


# num_experts_to_offload = [offload_per_layer - 1] * config.num_hidden_layers
# remaining = offload_config.offload_size - sum(num_experts_to_offload)
# # Randomly distribute the remaining among the elements
# while remaining > 0:
#     # Choose a random index
#     idx = np.random.randint(0, config.num_hidden_layers)
#     # Only add to the element if it's less than 5
#     if num_experts_to_offload[idx] < 5:
#         num_experts_to_offload[idx] += 1
#         remaining -= 1

# experts_to_offload = []
# for i, num_exp in enumerate(num_experts_to_offload):
#     experts_to_offload.append(np.random.choice(num_experts, num_exp, replace=False))


model, expert_cache = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=state_path,
    # experts_to_offload=experts_to_offload,
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# while True:
# print("User: ", end="")
# user_input = input()
# print("\n")

# user_input = "Tell me about India in 10 words."

with open("prompts.txt", "r") as f:
    prompts = f.read().splitlines()

for user_input in prompts:
    past_key_values = None
    sequence = None
    seq_len = 0
    print("=" * 50)
    print("Prompt: ", user_input)
    user_entry = dict(role="user", content=user_input)
    input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(
        device
    )

    attention_mask = torch.ones_like(input_ids)
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
    # get the total input and output sequence length
    seq_len = sum([len(seq) for seq in sequence])
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
    # print(f"Layer {layer_idx}: Hits: {hits}, Misses: {misses}")
    expert_hits[layer_idx] = eviction_group_info.expert_hits

<<<<<<< HEAD
print("Total hits: ", sum(layers2hits.values()))
print("Total misses: ", sum(layers2misses.values()))
print("Hit Rate: ", sum(layers2hits.values()) / (sum(layers2hits.values()) + sum(layers2misses.values())))
print(expert_hits)
=======
# print("Total hits: ", sum(layers2hits.values()))
# print("Total misses: ", sum(layers2misses.values()))
hits = sum(layers2hits.values())
misses = sum(layers2misses.values())
print(f"Hit ratio: {hits / (hits + misses)}({hits} hits, {misses} misses)")
>>>>>>> 6343388 (add dynamic expert selection)

experts_hits_modified = {}
for layer_idx, layer_expert_hits in expert_hits.items():
    for (_, expert_idx), hits in layer_expert_hits.items():
        if layer_idx not in experts_hits_modified:
            experts_hits_modified[layer_idx] = {}
        if expert_idx not in experts_hits_modified[layer_idx]:
            experts_hits_modified[layer_idx][expert_idx] = 0
        experts_hits_modified[layer_idx][expert_idx] += hits

# sort expert by expert idx
sorted_expert_hits = {}
for layer_idx, expert_hits in experts_hits_modified.items():
    sorted_expert_hits[layer_idx] = dict(sorted(expert_hits.items()))

# save expert hits
with open("expert_hits.json", "w") as f:
    json.dump(sorted_expert_hits, f, indent=4)
