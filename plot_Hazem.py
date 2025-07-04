import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os




def plot():
    path = "./Results_paper/"
    info = {}
    experimets = set()
    for folder in os.listdir(path):
        
        #remove Results_ from the folder name
        if folder.startswith("Results_"):
            folder_name = folder.replace("Results_", "")
            info[folder] = {}
            local_info = folder_name.split("_")
            if local_info[0] == "RNN":
                info[folder]["model"] = "RNN"
                info[folder]["states"] = ""
                info[folder]["env"] = local_info[1]
                if len(experimets) == 0:
                    for experiment in os.listdir(path + folder):
                        if not experiment.endswith("goal"):
                            experimets.add(experiment)
           
                info[folder]["model"] = "NRM"
                info[folder]["states"] = local_info[1]

                info[folder]["env"] = local_info[2]
    print(info)

    #now enter inside the folder Results_paper/Results_RNN_image_env
    print(experimets)
    print(len(experimets))


        

if __name__ == "__main__":
    plot()