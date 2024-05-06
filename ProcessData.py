import csv


CPU_PATHS     = ['C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu2CPU.txt', 'C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu3CPU.txt','C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu5CPU.txt']
MEMORY_PATHS  = ['C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu2MEMORY.txt','C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu3MEMORY.txt','C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu5MEMORY.txt']
POWER_PATHS   = ['C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu2POWER.txt','C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu3POWER.txt','C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu5POWER.txt']
LATENCY_PATHS = ['C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu2LATENCY.txt','C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu3LATENCY.txt','C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu5LATENCY.txt']
DISK_PATHS = ['C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu2STORAGE.txt','C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu3STORAGE.txt','C:/Users/Orin Claeys/Documents/MAP_DRL_Code/MAP_DRL/data_files/data_set1/rsu5STORAGE.txt']

with open(LATENCY_PATHS[0], 'r') as file:
    rsu2_latencys = file.readlines()
with open(LATENCY_PATHS[1], 'r') as file:
    rsu3_latencys = file.readlines()
with open(LATENCY_PATHS[2], 'r') as file:
    rsu5_latencys = file.readlines()

size = max(len(rsu2_latencys),len(rsu3_latencys),len(rsu5_latencys))

with open(CPU_PATHS[0], 'r') as file:
    rsu2_cpus = file.readlines()
with open(CPU_PATHS[1], 'r') as file:
    rsu3_cpus = file.readlines()
with open(CPU_PATHS[2], 'r') as file:
    rsu5_cpus = file.readlines()

with open(MEMORY_PATHS[0], 'r') as file:
    rsu2_mems = file.readlines()
with open(MEMORY_PATHS[1], 'r') as file:
    rsu3_mems = file.readlines()
with open(MEMORY_PATHS[2], 'r') as file:
    rsu5_mems = file.readlines()

with open(DISK_PATHS[0], 'r') as file:
    rsu2_disks = file.readlines()
with open(DISK_PATHS[1], 'r') as file:
    rsu3_disks = file.readlines()
with open(DISK_PATHS[2], 'r') as file:
    rsu5_disks = file.readlines()

with open(POWER_PATHS[0], 'r') as file:
    rsu2_pows = file.readlines()
with open(POWER_PATHS[1], 'r') as file:
    rsu3_pows = file.readlines()
with open(POWER_PATHS[2], 'r') as file:
    rsu5_pows = file.readlines()

for i in range(size):
    line_rsu2 = rsu2_latencys[i]
    line_rsu3 = rsu3_latencys[i]
    line_rsu5 = rsu5_latencys[i]
    line_values_2 = line_rsu2.split()
    line_values_3 = line_rsu3.split()
    line_values_5 = line_rsu5.split()
    t = float(line_values_2[0])
    rsu2_latency = line_values_2[1]
    rsu3_latency = line_values_3[1]
    rsu5_latency = line_values_5[1]

    for j in range(len(rsu2_cpus)):
        rsu2_cpu = rsu2_cpus[j].split()
        rsu2_mem = rsu2_mems[j].split()
        rsu2_pow = rsu2_pows[j].split()
        rsu2_disk = rsu2_disks[j].split()
        time = float(rsu2_cpu[0])
        if time>t-2 and time<t+2:
            rsu2_cpu_value = rsu2_cpu[1]
            rsu2_mem_value = rsu2_mem[1]
            rsu2_pow_value = rsu2_pow[1]
            rsu2_disk_value = rsu2_disk[1]
    
    for j in range(len(rsu3_cpus)):
        rsu3_cpu = rsu3_cpus[j].split()
        rsu3_mem = rsu3_mems[j].split()
        rsu3_pow = rsu3_pows[j].split()
        rsu3_disk = rsu3_disks[j].split()
        time = float(rsu3_cpu[0])
        if time>t-2 and time<t+2:
            rsu3_cpu_value = rsu3_cpu[1]
            rsu3_mem_value = rsu3_mem[1]
            rsu3_pow_value = rsu3_pow[1]
            rsu3_disk_value = rsu3_disk[1]

    for j in range(len(rsu5_cpus)):
        rsu5_cpu = rsu5_cpus[j].split()
        rsu5_mem = rsu5_mems[j].split()
        rsu5_pow = rsu5_pows[j].split()
        rsu5_disk = rsu5_disks[j].split()
        time = float(rsu5_cpu[0])
        if time>t-2 and time<t+2:
            rsu5_cpu_value = rsu5_cpu[1]
            rsu5_mem_value = rsu5_mem[1]
            rsu5_pow_value = rsu5_pow[1]
            rsu5_disk_value = rsu5_disk[1]

    state2 = [rsu2_cpu_value,rsu2_mem_value,rsu2_disk_value,rsu2_pow_value,rsu3_cpu_value,rsu3_mem_value,rsu3_disk_value,rsu3_pow_value,rsu5_cpu_value,rsu5_mem_value,rsu5_disk_value,rsu5_pow_value,0]
    state3 = [rsu2_cpu_value,rsu2_mem_value,rsu2_disk_value,rsu2_pow_value,rsu3_cpu_value,rsu3_mem_value,rsu3_disk_value,rsu3_pow_value,rsu5_cpu_value,rsu5_mem_value,rsu5_disk_value,rsu5_pow_value,1]
    state5 = [rsu2_cpu_value,rsu2_mem_value,rsu2_disk_value,rsu2_pow_value,rsu3_cpu_value,rsu3_mem_value,rsu3_disk_value,rsu3_pow_value,rsu5_cpu_value,rsu5_mem_value,rsu5_disk_value,rsu5_pow_value,2] 

    action = [rsu2_latency,rsu3_latency,rsu5_latency]

    file_path = "states.csv"
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(state2)
        writer.writerow(state3)
        writer.writerow(state5)
    file_path = "actions.csv"
    with open(file_path,mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(action)
        writer.writerow(action)
        writer.writerow(action)



    