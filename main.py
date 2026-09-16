from toyc import Cpu , GpuVulkan , bench , to_csv , batch_bench  
#This the code using to test 
# and generate resultes for generating chartes


# Test and gpu warmup

code = "1+2"

print ("Cpu:",Cpu.run(program = code, silent = True)) # this show the intiger as expected
print ("Gpu:",GpuVulkan.run(program = code, silent = True)) # this will show float not int because of how the gpu calcutae 


# Benching Loop using the build in bentch funtion on toyc 

# Single queue execution without batching


# Single bentch Cpu loop
for num in range(1 , 101): # sake of keping the data start from 1 insted of 0 and end in 999
    cpu_single_bench = bench(program=code,n=num,repeat=5,backend="cpu")
    to_csv(cpu_single_bench,"bench-csv/"+str(num)+"cpu_single_bench.csv")










# stopping the gpu from buring an going KABOOM (jk) it just causes a OOM
GpuVulkan.shutdown()