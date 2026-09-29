# Memory footprint of the foundation model

bytes_per_param = torch.tensor([], dtype=f_model.model.dtype).element_size()
theoretical_size = f_model.num_parameters * bytes_per_param

print("dtype:", f_model.model.dtype, "->", bytes_per_param, "bytes per parameter")
print("Theoretical model size: {:.2f} GB".format(theoretical_size / 1e9))
print("Memory footprint given by transformers: {:.2f} GB".format(f_model.model.get_memory_footprint() / 1e9))

if device.type == "cuda":
    print("Memory allocated on the GPU: {:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))
elif device.type == "mps":
    print("Memory allocated on the GPU (MPS): {:.2f} GB".format(torch.mps.current_allocated_memory() / 1e9))
