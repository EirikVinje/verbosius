import subprocess
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_gpu_temp():

    nvidia_smi = subprocess.check_output(["nvidia-smi", "-q", "-d", "TEMPERATURE"])
    nvidia_smi = nvidia_smi.decode()
    lines = nvidia_smi.split('\n')

    temps = []
    for line in lines:
        if "GPU Current Temp" in line:
            temps.append(line)

    temperature1 = temps[0].split(':')[-1].split('C')[0].strip()
    temperature2 = temps[1].split(':')[-1].split('C')[0].strip()

    return temperature1, temperature2

def get_gpu_mem():

    nvidia_smi = subprocess.check_output(["nvidia-smi", "-q", "-d", "MEMORY"])
    nvidia_smi = nvidia_smi.decode()
    lines = nvidia_smi.split('\n')

    mems = []
    for line in lines:
        if "Used" in line:
            mems.append(line)

    mem1 = mems[0].split(':')[-1].split('MiB')[0].strip()
    mem2 = mems[1].split(':')[-1].split('MiB')[0].strip()

    return mem1, mem2


def gpu_is_active():

    nvidia_smi_output = subprocess.check_output(["nvidia-smi"])
    nvidia_smi_output = nvidia_smi_output.decode()
    lines = nvidia_smi_output.split('\n')

    if "+----" in lines[1]:
        return True
    else:
        return False


def make_csv():

    if not os.path.exists("gpu_temp.csv"):
        df = pd.DataFrame({"gpu1": [], "gpu2": [], "mem1": [], "mem2": [], "time": []})
        df.to_csv("gpu_temp.csv", index=False)

    else:
        os.system("rm -rf gpu_temp.csv")
        df = pd.DataFrame({"gpu1": [], "gpu2": [], "mem1": [], "mem2": [], "time": []})
        df.to_csv("gpu_temp.csv", index=False)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_sleep", type=int, default=30, help="Time to sleep between each temperature check")
    args = parser.parse_args()

    ts = args.time_sleep

    make_csv()
    
    start = time.time()

    while gpu_is_active():

        print("GPU is active")

        temperature1, temperature2 = get_gpu_temp()
        mem1, mem2 = get_gpu_mem()

        if os.path.exists("gpu_temp.csv"):
            df = pd.read_csv("gpu_temp.csv")
            df = df.append({"gpu1": int(temperature1), 
                            "gpu2": int(temperature2),
                            "mem1": int(11016 - int(mem1)),
                            "mem2": int(11019 - int(mem2)), 
                            "time" : int(time.time() - start)}, 
                            ignore_index=True)
        
            df.to_csv("gpu_temp.csv", index=False)
        
        
        else:
            assert False, f"File does not exist"
        
        
        time.sleep(ts)

    print("GPU crashed")

    df = pd.read_csv("gpu_temp.csv")
    df.plot(x="time", y=["gpu1", "gpu2"])

    plt.savefig("gpu_temp.png")