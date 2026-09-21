#===================IMPORTS=================#
import os
import duckdb
import pandas as pd
import numpy as np

# Own custome importes made by spy1345a (author) or me who is woking on the code
from toyc import Cpu , GpuVulkan , bench , summarize , batch_bench  


#==============END OF IMPORTS=================#
#This the code using to test  and to generte the chatrs uusing plot.py file and the data is stored in the data folder as a database file called bench_data.db

# ==================CONFIG===================#
codes = ["1 + 2", "10 - 5", "5*10", "500/10"] # Equations used for testing (+,-,*,/), no zero division (that throws a run time error)
# will change the code -> Codes as a list to hold all the quiation for easire testing TO DO

n = 100_000 # times the equation is evaluated in a single batch (not repetitions — one continuous run of n evaluations)

r = 3 #Number of times to repeat the test to get he average of each 

os.makedirs("data", exist_ok=True) # create the data folder if it doesn't exist
data_path = "data/" # path to the database file
db_name = data_path + "bench_data" 

# Log scale steps from 1 to n — captures full curve with ~150 points instead of 1,000,000
ns = np.unique(np.logspace(0, np.log10(n), num=150, dtype=int))
ns = [int(x) for x in ns]

#=============DATA BASE CONFIG===============#
db = duckdb.connect(db_name+".db") # connect to the database file (will create if it doesn't exist)

def append_or_create(db, table, rows):
    df = pd.DataFrame(summarize(rows))  # collapse to summary first
    exists = db.execute(
        f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table}'"
    ).fetchone()[0]
    if exists:
        db.execute(f"INSERT INTO {table} SELECT * FROM df")
    else:
        db.execute(f"CREATE TABLE {table} AS SELECT * FROM df")

#==========END OF DATA BASE CONFIG============#


#===============END OF CONFIG================#

# Test and gpu warmup

code = codes[0]  # demo equation for the quick single runs below and warm up
print("WARM UP")
print ("Cpu:",Cpu.run(program = code, silent = True)) # this show the intiger as expected
print ("Gpu:",GpuVulkan.run(program = code, silent = True)) # this will show float not int because of how the gpu calcutae 
print("END OF WARM UP")

# Benching Loop using the build in bentch funtion on toyc 

#==========================BATCH BENCH==================#
print("\nSTART OF BATCH BENCH")

for code in codes:
    # Batch bentch Cpu loop
    for num in ns: # log scale steps from 1 to n
        cpu_batch_bench = batch_bench(program=code,n=num,repeat=r,backend="cpu",threads=4)
        append_or_create(db, "cpu_batch_bench", cpu_batch_bench)
        print(f"[cpu_batch] code='{code}' n={num}", flush=True)
    # Batch bentch Cpu loop
    for num in ns: # log scale steps from 1 to n
        gpu_batch_bench = batch_bench(program=code,n=num,repeat=r,backend="vulkan")
        append_or_create(db, "gpu_batch_bench", gpu_batch_bench)
        print(f"[gpu_batch] code='{code}' n={num}", flush=True)

print("END OF BATCH BENCH")
#===================END OF BATCH BENCH==================#



# Single queue execution without batching
#==========================SINGLE BENCH==================#
print("\nSTART OF SINGLE BENCH")

for code in codes:
    # Single bentch Cpu loop
    for num in ns: # log scale steps from 1 to n
        cpu_single_bench = bench(program=code,n=num,repeat=r,backend="cpu")
        append_or_create(db, "cpu_single_bench", cpu_single_bench)
        print(f"[cpu_single] code='{code}' n={num}", flush=True)

    # Single bench Gpu loop
    for num in ns: # log scale steps from 1 to n
        gpu_single_bench = bench(program=code,n=num,repeat=r,backend="vulkan")
        append_or_create(db, "gpu_single_bench", gpu_single_bench)
        print(f"[gpu_single] code='{code}' n={num}", flush=True)

print("END OF SINGLE BENCH")
#===================END OF SINGLE BENCH==================#


db.close()

# stopping the gpu from buring and going KABOOM (jk) it just causes a OOM if you over load it somehow idk how
GpuVulkan.shutdown()
