"""
We will profile attention
1. Its computational complexity with FLOPS
2. Its memory usage 
3. Its clock time 
Then we will collect the data for each metrics (FLOPs, memory usage, and clock time) for different input lengths 
"""
import torch
import torch.nn.functional as F 
import torch.autograd.profiler as profiler 
import psutil 
import time 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # for creating directories 
"""
First define a self-attention module. 
"""
class SelfAttention(torch.nn.Module): 
    # torch.nn.Module is the base class for all neural network modules in PyTorch 
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__() # super is used to call the parent class constructor
        self.query = torch.nn.Linear(embed_dim, embed_dim) 
        self.key = torch.nn.Linear(embed_dim, embed_dim) 
        self.value = torch.nn.Linear(embed_dim, embed_dim) 
    
    # our input x is a tensor of shape (batch_size, seq_len, embed_dim) 
    def forward(self,x): 
        # we will first compute the query, key, and value matrices 
        Q = self.query(x) 
        K = self.key(x) 
        V = self.value(x) 
        # we will compute the dot product of the query and key
        # Q and K have shape (batch_size, seq_len, embed_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) # transpose (-2,-1) means we are transposing the last two dimensions 
        # we will normalize the attention scores 
        attn_scores = F.softmax(attn_scores, dim=-1) 
        # we will multiply the attention scores with the value matrix 
        attn_output = torch.matmul(attn_scores, V) 
        return attn_output 

"""
1. To compute FLOPS, one can mannually count the operations or use Pytorch's profiler
"""
def profile_attention(model, input_data):
    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("self_attention"):
            model(input_data)
    flops = prof.key_averages().total_average().cpu_time_total  # Get total FLOPS
    mem_usage = memory_usage()
    time_taken = measure_wall_clock_time(model, input_data)
    return flops, mem_usage, time_taken


"""
2. To compute memory usage using torch.cuda.memory_allocated() and torch.cuda.memory_reserved()
"""

def memory_usage():
    return psutil.Process().memory_info().rss/1024**2 # return memory usage in MB 

# for gpu 
def gpu_memory_usage():
    return torch.cuda.memory_allocated()/1024**2, torch.cuda.memory_reserved()/1024**2 # return memory usage in MB 

"""
3. To measure the time, use the time module 
"""
def measure_wall_clock_time(model, input_data):
    start_time= time.time() 
    model(input_data) 
    end_time = time.time() 
    return end_time - start_time 

"""
4. Auxiliary functions for standard error 
"""
def standard_error(data):
    return np.std(data) / np.sqrt(len(data))


"""
Collect results: varying input length 
"""

embed_dim = 128
model = SelfAttention(embed_dim) 

# now we create input_lengths from 2^0 to 2^24 
# input_lengths from 10^1 to 10^4 in smaller steps
input_lengths = [2**i for i in range(3, 14)] 

# Run the experiment multiple times and collect results
n_runs = 20  # Number of runs to average over 
flops_list, mem_list, time_list = [], [], []

for length in input_lengths:
    flops_runs, mem_runs, time_runs = [], [], []
    for _ in range(n_runs):
        print(f"Running for input length {length} at the {_}th run")
        print("=============================================")
        input_data = torch.randn(1, length, embed_dim)
        flops, mem_usage, time_taken = profile_attention(model, input_data)
        flops_runs.append(flops)
        mem_runs.append(mem_usage)
        time_runs.append(time_taken)
    
    flops_list.append((np.mean(flops_runs), standard_error(flops_runs)))
    mem_list.append((np.mean(mem_runs), standard_error(mem_runs)))
    time_list.append((np.mean(time_runs), standard_error(time_runs)))


# Convert the data into a DataFrame for easier plotting with seaborn
data = pd.DataFrame({
    'Input Length': input_lengths,
    'FLOPS Mean': [flop[0] for flop in flops_list],
    'FLOPS SE': [flop[1] for flop in flops_list],
    'Memory Mean': [mem[0] for mem in mem_list],
    'Memory SE': [mem[1] for mem in mem_list],
    'Time Mean': [time[0] for time in time_list],
    'Time SE': [time[1] for time in time_list]
})

data['FLOPS Lower'] = data['FLOPS Mean'] - data['FLOPS SE']
data['FLOPS Upper'] = data['FLOPS Mean'] + data['FLOPS SE']

data['Memory Lower'] = data['Memory Mean'] - data['Memory SE']
data['Memory Upper'] = data['Memory Mean'] + data['Memory SE']

data['Time Lower'] = data['Time Mean'] - data['Time SE']
data['Time Upper'] = data['Time Mean'] + data['Time SE']


output_dir = "profiling_attention"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



""""
Now we make the plots 
"""
sns.lineplot(x='Input Length', y='FLOPS Mean', data=data)
plt.errorbar(data['Input Length'], data['FLOPS Mean'], yerr=data['FLOPS SE'], fmt='o', label='FLOPS')
plt.xscale('log')
plt.yscale('log')
plt.title("FLOPS vs Input Length")
plt.savefig(os.path.join(output_dir, 'flops_vs_input_length.png'))  # Save the plot in the folder
plt.clf()  # Clear the figure after saving

# Plot for Memory
sns.lineplot(x='Input Length', y='Memory Mean', data=data)
plt.errorbar(data['Input Length'], data['Memory Mean'], yerr=data['Memory SE'], fmt='o', label='Memory')
plt.xscale('log')
plt.yscale('log')
plt.title("Memory Usage vs Input Length")
plt.savefig(os.path.join(output_dir, 'memory_vs_input_length.png'))
plt.clf()

# Plot for Time
sns.lineplot(x='Input Length', y='Time Mean', data=data)
plt.errorbar(data['Input Length'], data['Time Mean'], yerr=data['Time SE'], fmt='o', label='Time')
plt.xscale('log')
plt.yscale('log')
plt.title("Wall Clock Time vs Input Length")
plt.savefig(os.path.join(output_dir, 'time_vs_input_length.png'))
plt.clf()


# Plot FLOPS with confidence band
plt.figure(figsize=(8, 6))
plt.plot(data['Input Length'], data['FLOPS Mean'], label='FLOPS Mean')
plt.fill_between(data['Input Length'], data['FLOPS Lower'], data['FLOPS Upper'], color='blue', alpha=0.3, label='Standard Error Band')

plt.xscale('log')
plt.yscale('log')
plt.title(f"FLOPS vs Input Length with Confidence Band with {n_runs} runs")
plt.xlabel('Input Length')
plt.ylabel('FLOPS')
plt.legend()

# Save the plot
plt.savefig(os.path.join(output_dir, 'flops_vs_input_length_with_band.png'))
plt.clf()


"""
Plots below have confidence bands 
"""
# Plot Memory with confidence band
plt.figure(figsize=(8, 6))
plt.plot(data['Input Length'], data['Memory Mean'], label='Memory Mean')
plt.fill_between(data['Input Length'], data['Memory Lower'], data['Memory Upper'], color='green', alpha=0.3, label='Standard Error Band')

plt.xscale('log')
plt.yscale('log')
plt.title(f"Memory Usage vs Input Length with Confidence Band with {n_runs} runs")
plt.xlabel('Input Length')
plt.ylabel('Memory Usage (MB)')
plt.legend()

# Save the plot
plt.savefig(os.path.join(output_dir, 'memory_vs_input_length_with_band.png'))
plt.clf()
# Plot Time with confidence band
plt.figure(figsize=(8, 6))
plt.plot(data['Input Length'], data['Time Mean'], label='Time Mean')
plt.fill_between(data['Input Length'], data['Time Lower'], data['Time Upper'], color='red', alpha=0.3, label='Standard Error Band')

plt.xscale('log')
plt.yscale('log')
plt.title(f"Wall Clock Time vs Input Length with Confidence Band with {n_runs} runs")
plt.xlabel('Input Length')
plt.ylabel('Wall Clock Time (seconds)')
plt.legend()

# Save the plot
plt.savefig(os.path.join(output_dir, 'time_vs_input_length_with_band.png'))
plt.clf()
