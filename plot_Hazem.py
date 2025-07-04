import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from tqdm import tqdm


def plot():
    PATH_BASE = "./Results_paper/"
    info = {}
    experimets = set()
    for folder in os.listdir(PATH_BASE):

        # remove Results_ from the folder name
        if folder.startswith("Results_"):
            folder_name = folder
            info[folder] = {}
            local_info = folder_name.split("_")
            if local_info[1] == "RNN":
                info[folder]["model"] = "RNN"
                info[folder]["states"] = ""
                info[folder]["env"] = local_info[2]
                if len(experimets) == 0:
                    for experiment in os.listdir(PATH_BASE + folder):
                        if not experiment.endswith("goal"):
                            experimets.add(experiment)
            else:

                info[folder]["model"] = "NRM"
                info[folder]["states"] = local_info[2]
                info[folder]["env"] = local_info[3]
    print(info)

    # now enter inside the folder Results_paper/Results_RNN_image_env
    experiments = list(experimets)
    print(len(experimets))

    NUM_EXP = 5

    for env in ["image", "map"]:
        if env == "image":
            save_path = "./Figures/Image_env/"
        else:
            save_path = "./Figures/Map_env/"
        for experiment in tqdm(experiments, desc=f"Processing {env} enviroment"):
            folder_to_consider = []
            for folder in info.keys():
                if info[folder]["env"] == env:
                    folder_to_consider.append(folder)
            results_experimets = []
            
            for folder in tqdm(folder_to_consider, leave=False, desc=f"Processing {env} enviroment, {experiment} experiment"):
                path = os.path.join(PATH_BASE, folder, experiment)
                
                #there are NUM_EXP files in the folder called train_rewards_<experiment>.txt, train_rewards_<experiment>.txt, ...
                #avarage them using seaborn and plot them
                model_Experiment = []
                for i in tqdm(range(NUM_EXP), leave=False, desc=f"Processing {folder}"):
                    
                    file_path = os.path.join(path, f"train_rewards_{i}.txt")
                    with open(file_path, "r") as f:
                        lines = f.readlines()
                    lines = [float(line.strip()) for line in lines]
                    lines = np.convolve(lines, np.ones(100)/100, mode='valid')
                    results_experimets.append(lines)
                model_Experiment.append(results_experimets)
            
            # now model_Experiment is a list of lists, where each list is the results of 5 same experiments. Plot them and have each model have its line
            #use seaborn to plot the results
            plt.figure(figsize=(10, 6))
            for i, results in enumerate(model_Experiment):
                label = f"{info[folder_to_consider[i]]['model']} - {info[folder_to_consider[i]]['states']} States"
                print(label)
                
                # Convert to long-form DataFrame for Seaborn
                df = pd.DataFrame(results).T  # Transpose so each column is one run
                df = df.melt(var_name="Run", value_name="Reward")
                df["Step"] = df.index % len(results[0])
                
                sns.lineplot(data=df, x="Step", y="Reward", label=label, errorbar='sd')

            plt.title(f"Training Rewards for {experiment} in {env} environment")
            plt.xlabel("Training Steps")
            plt.ylabel("Average Reward")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(save_path, f"{experiment}_{env}_training_rewards.png"))
            plt.close()

            

                        
                    

            print("-------------------")


def plot_GPT():

    PATH_BASE = "./Results_paper/"
    info = {}
    experimets = set()

    # Collect info from folder names
    for folder in os.listdir(PATH_BASE):
        if folder.startswith("Results_"):
            folder_name = folder
            info[folder] = {}
            local_info = folder_name.split("_")
            if local_info[1] == "RNN":
                info[folder]["model"] = "RNN"
                info[folder]["states"] = ""
                info[folder]["env"] = local_info[2]
                if len(experimets) == 0:
                    for experiment in os.listdir(os.path.join(PATH_BASE, folder)):
                        if not experiment.endswith("goal"):
                            experimets.add(experiment)
            else:
                info[folder]["model"] = "NRM"
                info[folder]["states"] = local_info[2]  # Distinguish by state
                info[folder]["env"] = local_info[3]

    experiments = list(experimets)
    NUM_EXP = 5

    # Prepare color palette with consistent model names
    model_labels = set()
    for val in info.values():
        if val["model"] == "RNN":
            model_labels.add("RNN")
        else:
            model_labels.add(f"NRM {val['states']}")

    model_labels = sorted(model_labels)
    palette = dict(zip(model_labels, sns.color_palette( n_colors=len(model_labels))))

    for env in ["image", "map"]:
        save_path = f"./Figures/{env.capitalize()}_env/"
        os.makedirs(save_path, exist_ok=True)

        for experiment in tqdm(experiments, desc=f"Processing {env} environment"):
            folder_to_consider = [folder for folder in info if info[folder]["env"] == env]
            
            all_data = []

            for folder in tqdm(folder_to_consider, leave=False, desc=f"{experiment} in {env}"):
                model = info[folder]["model"]
                state = info[folder]["states"]
                model_label = model if model == "RNN" else f"NRM {state}"

                path = os.path.join(PATH_BASE, folder, experiment)
                
                for i in range(NUM_EXP):
                    file_path = os.path.join(path, f"train_rewards_{i}.txt")
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, "r") as f:
                        lines = [float(line.strip()) for line in f]
                    
                    smoothed = np.convolve(lines, np.ones(100) / 100, mode="valid")
                    for step, value in enumerate(smoothed):
                        all_data.append({
                            "Step": step,
                            "Reward": value,
                            "Model": model_label
                        })

            if not all_data:
                print(f"No data found for experiment: {experiment}")
                continue

            df = pd.DataFrame(all_data)
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x="Step", y="Reward", hue="Model", palette=palette, errorbar="sd")

            plt.title(f"{env[0].upper()}{env[1:]} Enviroment - Task {experiment.split(':')[0][-1]}")
            plt.xlabel("Step")
            plt.ylabel("Rewards")
            plt.grid(False)
            plt.tight_layout()
            #remove the legend
            plt.legend(title="Model", loc='upper left', bbox_to_anchor=(1, 1))
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(os.path.join(save_path, f"{experiment}.png"))
            plt.close()
                    



if __name__ == "__main__":
    plot_GPT()
