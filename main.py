from toyc import GpuVulkan, Cpu
from toyc.bench import bench, batch_bench, summarize, to_csv


code = "1+2"

cpu_result_single = Cpu.run(program="program.toy", silent=True)
print("cpu:", cpu_result_single)

gpu_result_single = GpuVulkan.run(program=code, silent=True)

print("gpu:",gpu_result_single)


GpuVulkan.shutdown()
