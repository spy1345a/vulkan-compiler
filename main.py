from toyc import Cpu , GpuVulkan , bench , to_csv , batch_bench  
#This the code using to test 
# and generate resultes for generating chartes
# ==================CONFIG===================
n = 100 # times the equation is evaluated in a single batch (not repetitions — one continuous run of n evaluations)

r = 5 #Number of times to repeat the test


#===============END OF CONFIG================

# Test and gpu warmup

code = "1+2"

print ("Cpu:",Cpu.run(program = code, silent = True)) # this show the intiger as expected
print ("Gpu:",GpuVulkan.run(program = code, silent = True)) # this will show float not int because of how the gpu calcutae 


# Benching Loop using the build in bentch funtion on toyc 

# Single queue execution without batching
cpu_single_bench = []
gpu_single_bench = []

# Single bentch Cpu loop
for num in range(1 , n+1): # start from 1 instead of 0, end at n (inclusive)
    cpu_single_bench += bench(program=code,n=num,repeat=r,backend="cpu")
    
to_csv(cpu_single_bench,"bench-csv/cpu_single_bench.csv")

# Single bench Gpu loop
for num in range(1 , n+1): # start from 1 instead of 0, end at n (inclusive)
    gpu_single_bench += bench(program=code,n=num,repeat=r,backend="vulkan")
    
to_csv(gpu_single_bench,"bench-csv/gpu_single_bench.csv")









# stopping the gpu from buring an going KABOOM (jk) it just causes a OOM
GpuVulkan.shutdown()