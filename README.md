vLLM performance analysis

# Goals

- Learn about vllm internals, and performance thereof
- Explore, experimentally, the elmination of startup delay, thus improving availability

# Model memory copy optimization

This code is lifted, paritally, from vllm so that I can experiment without all of vllm on board.

Purpose of `bench.py`:

- Performs cache reconstruction on a model from the hub identified by repo_id.
- Maps the model into CPU memory using `safetensors` mmap().
- Override some behaviors of torch.tensor's bytestorage (which is `torch.Storage`).

The override in part 3 of this experiment involves interjecting [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) with `safetensors`, and `pytorch`.

Explore:

- [cudaHostRegister](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8d5c17670f16ac4fc8fcb4181cb490c): Map GPU virtual memory to CPU virtual memory
- [cudaMallocManaged](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8d5c17670f16ac4fc8fcb4181cb490c): Map GPU virtual memory for memory oversubscribe. This may enable the GPU's DMA to transfer memory, instead of the CPU copying memory.
- [cudaMemcpyAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79): CUDA is a client/server system. In this case, use a CUDA stream on the server (GPU control chip) to DMA information from host to memory.

- There are other areas to explore. My itch is focused on reducing startup time, because the 1-2 minute starts
  are frustratingly long, for small models, served 100% from GPU memory.

# Current Results

Shows part 1, part 2. The exploration of  the experiment has not starteed. To destroy the cache, I `rm -rf $HOME/.cache/huggingface/hub` **WARNING** this is partially destructive.

This maps the full 67.68 GiB into the CPU address space.

No cache reconstruction. 70B model mapped into CPU address space.

```console
(v-safe2) sdake@a40x2:~/safe2$ python bench.py
model mapped size (approx.)=67.68 GiB
time to reference_model_by_repo=1572.24 ms
```


Full serial cache reconstruction, 70B model mapped into CPU address space.

```console
(v-safe2) sdake@a40x2:~/safe2$ python bench.py
model.safetensors.index.json: 100%|██████████████████████████████████████████████| 109k/109k [00:00<00:00, 996kB/s]
model-00003-of-00015.safetensors: 100%|████████████████████████████████████████| 4.90G/4.90G [00:40<00:00, 122MB/s]
model-00006-of-00015.safetensors: 100%|████████████████████████████████████████| 4.98G/4.98G [00:40<00:00, 124MB/s]
model-00004-of-00015.safetensors: 100%|████████████████████████████████████████| 4.90G/4.90G [00:40<00:00, 122MB/s]
model-00008-of-00015.safetensors: 100%|████████████████████████████████████████| 4.90G/4.90G [00:39<00:00, 125MB/s]
model-00015-of-00015.safetensors: 100%|████████████████████████████████████████| 3.81G/3.81G [00:30<00:00, 125MB/s]
model-00002-of-00015.safetensors: 100%|████████████████████████████████████████| 4.98G/4.98G [00:18<00:00, 265MB/s]
model-00001-of-00015.safetensors: 100%|████████████████████████████████████████| 4.82G/4.82G [00:40<00:00, 119MB/s]
model-00005-of-00015.safetensors: 100%|████████████████████████████████████████| 4.90G/4.90G [00:38<00:00, 127MB/s]
model-00007-of-00015.safetensors: 100%|████████████████████████████████████████| 4.90G/4.90G [00:40<00:00, 121MB/s]
model-00012-of-00015.safetensors: 100%|████████████████████████████████████████| 4.90G/4.90G [00:41<00:00, 119MB/s]
model-00009-of-00015.safetensors: 100%|████████████████████████████████████████| 4.90G/4.90G [00:39<00:00, 125MB/s]
model-00014-of-00015.safetensors: 100%|████████████████████████████████████████| 4.98G/4.98G [00:40<00:00, 122MB/s]
model-00013-of-00015.safetensors: 100%|████████████████████████████████████████| 4.90G/4.90G [00:40<00:00, 122MB/s]
model-00011-of-00015.safetensors: 100%|████████████████████████████████████████| 4.90G/4.90G [00:40<00:00, 122MB/s]
model-00010-of-00015.safetensors: 100%|████████████████████████████████████████| 4.98G/4.98G [00:37<00:00, 134MB/s]
model mapped size (approx.)=67.68 GiB
time to reference_model_by_repo=571159.87 ms
```

Full cache reconstruction (parallel), 70B model mapped into CPU address space.

```console
code not quite complete.
```


# Measuring startup dealy

Startup expense in vllm 0.5.4 with 2 A40 (48gb ram), 768GB ram, 32 cores:

```console
12 seconds to load model
14 seconds to execute_model
	2 seconds to calculate hidden_or_intermediate_states
	12 seconds to calculate logits
Service becomes available.
```

Detailed timestamp analysis:

```console
INFO 08-26 20:37:53 model_runner.py:720] Starting to load model neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8...
INFO 08-26 20:37:53 weight_utils.py:225] Using model weights format ['*.safetensors']
INFO 08-26 20:37:53 llama.py:467] load llama3.1 weights
INFO 08-26 20:38:05 llama.py:522] done load llama3.1 weights
INFO 08-26 20:38:05 model_runner.py:734] Loading model weights took 33.8807 GB
INFO 08-26 20:38:05 model_runner.py:796] load_model done
INFO 08-26 20:38:05 distributed_gpu_executor.py:38] run_workers
INFO 08-26 20:38:05 model_runner.py:1281] prepare_model_input
INFO 08-26 20:38:05 model_runner.py:1295] done prepare_model_input
INFO 08-26 20:38:05 model_runner.py:1309] execute_model
INFO 08-26 20:38:05 model_runner.py:1377] calculate hidden_or_intermediate_states
INFO 08-26 20:38:07 model_runner.py:1387] done calculate hidden_or_intermediate_states
INFO 08-26 20:38:07 model_runner.py:1393] calculate logits
INFO 08-26 20:38:19 model_runner.py:1396] logits is tensor([[ 4.3750,  3.4219,  2.5938,  ..., -8.0000, -8.0000, -8.0000],
INFO 08-26 20:38:19 model_runner.py:1420] done execute_model
INFO 08-26 20:38:19 distributed_gpu_executor.py:40] done run_workers num_blocks=[(2064, 1638), (2076, 1638)]
INFO 08-26 20:38:19 distributed_gpu_executor.py:58] # GPU blocks: 2064, # CPU blocks: 1638
WARNING 08-26 20:38:23 serving_embedding.py:171] embedding_mode is False. Embedding API will not work.
```

# TODO

- Examine envs.VLLM_TEST_DYNAMO_GRAPH_CAPTURE: which runs torch.compile()

# Code mapping

These were in my notes, not sure what their value is.T
```console
720:model_runner.py:load_model()
	Calls model_runner.py:get_model()
331:loader.py:load_model()
	* Creates a bound method bound method LlamaForCausalLM.load_weights of LlamaForCausalLM
387:weight_utils.py:safetensors_weights_iterator()
	* loads the weights with the safe tensors library
340:llama.py:LlamaForCausalLM()
457:lama.py:load_weights()
```

