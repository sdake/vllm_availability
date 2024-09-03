import os
import time
from typing import List, Dict, Tuple, Callable
import torch
from safetensors.torch import load_file, storage_size
from safetensors import safe_open
from transformers import AutoModel
from huggingface_hub import snapshot_download, hf_hub_download
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
import json


def to_gib(i: int) -> float:
    return i / 1024**3


def to_ms(i: int) -> float:
    return i / 1e6


###
#
# Store metadata in safetensor

metadata: Dict[str, str] = {
    "They who created this example": "Steven Dake",
    "model_name": "Llama 70B",
    "model_version": "1.0",
    "num_layers": "70",
    "precision": "w8a8",
    "tensor_parallelism": "2",
    "pipeline_parallelism": "2",
    "vocab_size": "50257",
    "seq_length": "2048",
    "creation_date": "2024-09-02",
    "author": "Steven Dake",
    "description": "Llama 70B model using w8a8 precision with tensor and pipeline parallelism",
}


###
#
# Perform cache reconstruction for safetensors by repo-id. Then reference the model by mapping the model
# into the CPUs address space. Do not, however, map or copy into the GPU address space or memory.
#
# This function is unneccessary rework of the below permalink. However, I needed to implement it myself
# outside of vllm so that I can understand the safetensor CPU mapping and map them into the GPU address space
# with cudaHostAlloc:
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8d5c17670f16ac4fc8fcb4181cb490c
#
# Reference of original safetensors weight management:
# https://github.com/vllm-project/vllm/blob/dd2a6a82e3f41b4673b1dbb24b2e99230ea96981/vllm/model_executor/model_loader/weight_utils.py#L252

def reference_model_by_repo(repo_id: str) -> None:
    size = 0

    safetensors_index_path = hf_hub_download(repo_id=repo_id, filename=SAFE_WEIGHTS_INDEX_NAME)

    with open(safetensors_index_path) as safetensors_index_ref:
        weight_map = json.load(safetensors_index_ref)["weight_map"]

    safetensors_path = os.path.dirname(safetensors_index_path)

    # I need a list of safetensor files. The association betwen the layer_name (key) and the tensor
    # filename (value) does not need to be stored. In a different step, the layers are loaded from
    # a unified hash map. Therefore, set() is used to store the safetensors files.

    safetensor_files = set(weight_map.values())
    for safetensor_file in safetensor_files:
        hf_hub_download(repo_id=repo_id, filename=safetensor_file)

    safetensor_file_paths = {f"{safetensors_path}/{safetensor_file}" for safetensor_file in safetensor_files}

    state_dict: Dict[str, torch.Tensor] = {}

    # Merge all keys(tensor name), values (torch.Tensor) into one dictionary
    for safetensor_file_path in safetensor_file_paths:
        file_dict = load_file(safetensor_file_path)
        if isinstance(file_dict, dict):
            for key, tensor in file_dict.items():
                if isinstance(tensor, torch.Tensor):
                    state_dict[key] = tensor
                else:
                    raise TypeError(f"Expected torch.Tensor, got {type(tensor)} for key {key}")
        else:
            raise TypeError(f"Expected dict from load_file, got {type(file_dict)}")

    size = sum(storage_size(v) for v in state_dict.values())
    print(f"model mapped size (approx.)={to_gib(size):.2f} GiB")


###
#
# Reference a safetensor of 15 safetensor files into CPU memory, but do not copy to PGU
# This can be removed once in git archive3. The current implementation is "complete"
# while this one has troubling hardcodes.

def ref_original() -> None:
    size = 0
    #state_dict: List[Dict[str, torch.Tensor]] = [{} for _ in range(15)]

    repo_id = 'neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8'
    safetensors_index_path = hf_hub_download(repo_id=repo_id, filename=SAFE_WEIGHTS_INDEX_NAME)
    safetensors_index_ref = open(safetensors_index_path)
    weight_map = json.load(safetensors_index_ref)["weight_map"]
    safetensors_path = os.path.dirname(safetensors_index_path)
    print(f'{safetensors_index_path}')

    safetensor_files = set(weight_map.values())
    for safetensor_file in safetensor_files:
        hf_hub_download(repo_id=repo_id, filename=safetensor_file)

    # I need a list of safetensor files. The association betwen the layer_name (key) and the tensor
    # filename (value) does not need to be stored. In a different step, the layers are loaded from
    # a unified hash map. Therefore, set() is used to store the safetensors files.

    safetensor_files = {f'{safetensors_path}/{safetensor_file}' for safetensor_file in safetensor_files.values()}
    print(f'safetensor_files={safetensor_files}')

#        repo_id=model_id, revision=revision, filename="pytorch_model.bin", token=token, cache_dir=folder
#    )
#    sf_name = "model.safetensors"
#    sf_filename = os.path.join(folder, sf_name)
#    convert_file(pt_filename, sf_filename, discard_names)
#    operations = [CommitOperationAdd(path_in_repo=sf_name, path_or_fileobj=sf_filename)]
#    errors: List[Tuple[str, "Exception"]] = []
#    return operations, errors


#    model = AutoModel.from_pretrained(model_name, revision='main', torch_dtype=torch.int8, cache_dir='/home/sdake/.cache/huggingface')
    
    # Assuming model saving to the expected directory
#    model.save_pretrained('/home/sdake/.cache/huggingface/hub/models--neuralmagic--Meta-Llama-3.1-70B-Instruct-quantized.w8a8')

#    for i in range(1, 15):
#        file_path = f'/home/sdake/.cache/huggingface/hub/models--neuralmagic--Meta-Llama-3.1-70B-Instruct-quantized.w8a8/snapshots/f3ebf1a9813986f26ec6984f5700281a9f6d8a54/model-000{i:02d}-of-00015.safetensors'
#        state_dict[i] = load_file(file_path)

 #       size += sum(storage_size(v) for v in state_dict[i].values())
    
 #   print(f'model size={to_gib(size):.2f} GiB')

###
#
a Reference a consolidated safetensor by mapping into CPU address space, but not GPU address space.

def ref_consolidated_one_way() -> None:
    file_path = '/home/sdake/safetensors/meta70b-a8w8.safetensors'
    state_dict = load_file(file_path)

    size = sum(storage_size(v) for v in state_dict.values())
    print(f'model size={to_gib(size):.2f} GiB')

###
#
# Demonstrates safetensor int3eraction with model metadeata

def display_metadata() -> None:
    file_path = '/home/sdake/safetensors/meta70b-a8w8.safetensors'
    with safe_open(file_path, framework='pt') as f:
        metadata = f.metadata()
    print(f'metadata={metadata}')

###
#
# Experiment using to('cuda:0') and to('cuda:1') to measure model laod time on two A40s. (11 seconds)

def ref_consolidated_two_ways() -> None:
    file_paths: List[Tuple[str, str]] = [
        ('/home/sdake/safetensors/meta70b-1-of-2-a8w8.safetensors', 'cuda:0'),
        ('/home/sdake/safetensors/meta70b-2-of-2-a8w8.safetensors', 'cuda:1')
    ]

    for file_path, device in file_paths:
        state_dict = load_file(file_path, device=device)
        size = sum(storage_size(v) for v in state_dict.values())
        print(f'{device} memory {to_gib(size):.2f} GiB')

###
#
# Consolidate the 15 tensors into one safetensor file. This is archival, once in git will remove.

def consolidate_one_way() -> None:
    new_state_dict: Dict[str, torch.Tensor] = {}
    for i in range(15):
        for k, v in state_dict[i].items():
            new_state_dict[k] = v

    save_file(new_state_dict, '/home/sdake/safetensors/meta70b-a8w8.safetensors', metadata=metadata)

###
#
# Consolidate the 15 tensors into two safetensor files. This is archival, once in git will remove.

def consolidate_two_ways() -> None:
    file_ranges: List[Tuple[int, int, str]] = [
        (0, 8, '/home/sdake/safetensors/meta70b-1-of-2-a8w8.safetensors'),
        (9, 15, '/home/sdake/safetensors/meta70b-2-of-2-a8w8.safetensors')
    ]

    for start, end, file_path in file_ranges:
        new_state_dict: Dict[str, torch.Tensor] = {}
        for i in range(start, end):
            for k, v in state_dict[i].items():
                new_state_dict[k] = v
        save_file(new_state_dict, file_path)

if __name__ == "__main__":
    functions: List[Tuple[str, Callable[[], None]]] = [
        ("reference_model_by_repo", reference_model_by_repo),
    ]
#        ("consolidate_one_way", consolidate_one_way),
#        ("consolidate_two_ways", consolidate_two_ways),
#        ("display_metadata", display_metadata),
#        ("ref_consolidated_one_way", ref_consolidated_one_way),
#        ("ref_consolidated_two_ways", ref_consolidated_two_ways)
#    ]

    for name, func in functions:
        start = time.perf_counter_ns()
        repo_id = "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8"
        func(repo_id)
        end = time.perf_counter_ns()
        print(f'time to {name}={to_ms(end-start):.2f} ms')
