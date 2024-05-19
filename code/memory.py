'''
Helpers we used to manage CUDA memory
'''
def ram_used():
    memory_bytes = torch.cuda.memory_allocated()
    memory_megabytes = memory_bytes / (1024 ** 2)
    print(f"Model is using approximately {memory_megabytes:.2f} MB of GPU memory.")
    
def memory_usage(tensor):
    element_size = tensor.element_size()  
    num_elements = tensor.numel() 
    total_memory_bytes = num_elements * element_size
    total_memory_mb = total_memory_bytes / (1024 ** 2) 

    print(f"Total memory usage of tensor: {total_memory_mb} MB")