#===================IMPORTS=================#
import duckdb
import pandas as pd

# Own custome importes made by spy1345a (author) or me who is woking on the code
from toyc import Cpu , GpuVulkan , bench , to_csv , batch_bench  


#==============END OF IMPORTS=================#
#This the code using to test 
# and generate resultes for generating chartes

# TO DO ADD DUCKDB AND PERSINTANC STOREGE , CSV FILES ARE USING TOO MUCH SPACE WE NNED TO ADD DUCK DB FOR HANDLING THE BENCHMARK STORE OR MY PC WILL CRY 

# ==================CONFIG===================#
codes = ["1 + 2", "10 - 5", "5*10", "500/10"] # Equations used for testing (+,-,*,/), no zero division (that throws a run time error)
# will change the code -> Codes as a list to hold all the quiation for easire testing TO DO

n = 1000 # times the equation is evaluated in a single batch (not repetitions — one continuous run of n evaluations)

r = 3 #Number of times to repeat the test to get he average of each 

#=============DATA BASE CONFIG===============#
db = duckdb.connect("data/data_bench_data.db")

def append_or_create(db, table, rows):
    df = pd.DataFrame(rows) # rows is a list of dicts returned by bench()
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

# Single queue execution without batching

#==========================SINGLE BENCH==================#
print("\nSTART OF SINGLE BENCH")

for code in codes:
    # Single bentch Cpu loop
    for num in range(1 , n+1): # start from 1 instead of 0, end at n (inclusive)
        cpu_single_bench = bench(program=code,n=num,repeat=r,backend="cpu")
        append_or_create(db, "cpu_single_bench", cpu_single_bench)

    # Single bench Gpu loop
    for num in range(1 , n+1): # start from 1 instead of 0, end at n (inclusive)
        gpu_single_bench = bench(program=code,n=num,repeat=r,backend="vulkan")
        append_or_create(db, "gpu_single_bench", gpu_single_bench)

print("END OF SINGLE BENCH")
#===================END OF SINGLE BENCH==================#


#==========================BATCH BENCH==================#
print("\n START OF BATCH BENCH")

for code in codes:
    for num in range(1 , n+1): # start from 1 instead of 0, end at n (inclusive)
        cpu_batch_bench = batch_bench(program=code,n=num,repeat=r,backend="cpu",threads=1)
        append_or_create(db, "cpu_batch_bench", cpu_batch_bench)

    for num in range(1 , n+1): # start from 1 instead of 0, end at n (inclusive)
        gpu_batch_bench = batch_bench(program=code,n=num,repeat=r,backend="vulkan")
        append_or_create(db, "gpu_batch_bench", gpu_batch_bench)

print("END OF BATCH BENCH")
#===================END OF BATCH BENCH==================#

db.close()
# stopping the gpu from buring and going KABOOM (jk) it just causes a OOM if you over load it somehow idk how
GpuVulkan.shutdown()