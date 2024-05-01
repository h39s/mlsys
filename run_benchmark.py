import sys

sys.path.append("./")
import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from IPython.display import clear_output
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging as hf_logging
import time
from tqdm import tqdm
from src.build_model import OffloadConfig, QuantConfig, build_model

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
state_path = "/home/amangupt/random/mixtral-offloading/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
benchmark_prompts = "benchmark_prompts.txt"

def read_prompt(file_path):
    with open(file_path, "r") as f:
        prompts = f.readlines()
    return prompts

all_prompts = read_prompt(benchmark_prompts)
all_prompts = all_prompts[:2]
# print(read_prompt(benchmark_prompts))


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


model, expert_cache_obj = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=state_path,
)

from transformers import TextStreamer


tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
past_key_values = None
sequence = None
total_time = []
total_num_tokens = []

seq_len = 0
for i in tqdm(range(len(all_prompts))):
  start = time.time()
  print("User: ", end="")
  user_input = all_prompts[i]
  print(user_input)
  print("\n")

  user_entry = dict(role="user", content=user_input)
  input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

  if past_key_values is None:
    attention_mask = torch.ones_like(input_ids)
  else:
    seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
    attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

  print("Mixtral: ", end="")
  result = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    streamer=streamer,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_hidden_states=True,
  )
  print("\n")
  sequence = result["sequences"]
  past_key_values = result["past_key_values"]
  end = time.time()
  total_time.append(end - start)
  seq_len = sum([len(seq) for seq in sequence])
  total_num_tokens.append(seq_len)

# CHANGE FILENAME HERE
filename = "Initial_attempt"

# make a track_results/logs/{filename}.txt file

log_file = open(f"track_results/logs/{filename}.txt", "w")
dump_data_file = open(f"track_results/data/{filename}.json", "w")


print("TIME BENCHMARKS", file=log_file)
print(f"Total time taken: {sum(total_time)} seconds", file=log_file)
print(f"Total number of tokens generated: {sum(total_num_tokens)}", file=log_file)
print(f"Average token per second: {sum(total_num_tokens)/sum(total_time)}", file=log_file)
print('\n\n\n', file=log_file)

print("HIT RATE BENCHMARKS", file=log_file)
data_hits = {}

for k in expert_cache_obj.group_infos:
    data_hits[k] = expert_cache_obj.group_infos[k].expert_counts
# print(data_hits)
# print overall hit rate and hit rate per layer
overall_hits = 0
overall_misses = 0
for layer in data_hits:
    tot_calls = 0
    tot_hits = 0
    # print(data_hits[layer])
    for exp in data_hits[layer]:
        tot_calls += data_hits[layer][exp][0]
        tot_hits += data_hits[layer][exp][1]
    # print(tot_hits, tot_calls)
    overall_hits += tot_hits
    overall_misses += tot_calls - tot_hits
    print(f"Layer {layer}: Hit rate = {tot_hits/tot_calls}", file=log_file)

print(f"Overall hit rate = {overall_hits/(overall_hits + overall_misses)}", file=log_file)


# dump data_hits, total_time, total_num_tokens to a json file
import json
all_stats = {"data_hits": data_hits, "total_time": total_time, "total_num_tokens": total_num_tokens}
json.dump(all_stats, dump_data_file)