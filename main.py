import os
import torch
from torch.nn import functional as F
import argparse
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def trim_linear(gate_layer,kept_experts):
    original_shape = gate_layer.weight.shape
    new_weight = torch.zeros(len(kept_experts), original_shape[1])
    new_weight[:, :] = gate_layer.weight[kept_experts, :]
    gate_layer.weight = nn.Parameter(new_weight)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default="/workspace/text-generation-webui2/models/mistralai_Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument('--target_dir', default="/src/models/mistralai_Mixtral-6x7B-Instruct-v0.1")
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--kept_experts', nargs='+', type=int, default=[0,2,4,5,6,7])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    model_id = args.model_id
    target_dir = args.target_dir
    load_in_8bit = args.load_in_8bit
    kept_experts = args.kept_experts

    n_experts = len(kept_experts)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if load_in_8bit:
        print("loading in 8bit")
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", trust_remote_code=False,load_in_8bit=True)
    else:
        print("Not loading 8bit")
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", trust_remote_code=False)
    config = AutoConfig.from_pretrained(model_id)
    config.num_local_experts = n_experts

    for i,layer in enumerate(model.model.layers):
        layer.block_sparse_moe.num_experts = n_experts
        trim_linear(layer.block_sparse_moe.gate,kept_experts)
        new_experts = nn.ModuleList([layer.block_sparse_moe.experts[i] for i in kept_experts])
        layer.block_sparse_moe.experts = new_experts

    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)

    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    config.save_pretrained(target_dir)