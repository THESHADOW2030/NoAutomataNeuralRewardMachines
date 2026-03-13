import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from tqdm import tqdm



PATH_0 = "Results_0/"


def get_training_rewards(path):
    with open(os.path.join(path, "train_rewards_0.txt"), "r") as f:
        rewards = f.read().splitlines()
    rewards = [float(r) for r in rewards]
    return rewards

def get_sequence_accuracies(path):
    with open(os.path.join(path, "sequence_classification_accuracy_0.txt"), "r") as f:
        accuracies = f.read().splitlines()
    accuracies = [float(a) for a in accuracies]
    return accuracies


def smooth_from_list(data, window_size):
    smoothed = []
    mean = 0
    for i, line in enumerate(data):
        if len(smoothed)<= window_size:
            mean += line
            smoothed.append(mean / (i + 1))
        else:
            mean += line - data[i - window_size]
            smoothed.append(mean / window_size)
    return smoothed


if __name__ == "__main__":

    rows = []

    for folder in os.listdir(PATH_0):
        print(f"Folder {folder}")
        task_number = folder.split("task")[1][0]
        path = os.path.join(PATH_0, folder)

        for model_type in os.listdir(path):

            if model_type.startswith("rnn"):
                for tmp in os.listdir(os.path.join(path, model_type)):

                    path_rnn = os.path.join(path, model_type, tmp)
                    rewards_rnn = get_training_rewards(path_rnn)
                    smoothed_rewards_rnn = smooth_from_list(rewards_rnn, 400)

                    rows.append({
                        "Task": f"Task {task_number}",
                        "Model": "RNN",
                        "State Number": None,
                        "Symbol Number": None,
                        "Smoothed Accuracy": None,
                        "Smoothed Reward": smoothed_rewards_rnn
                    })

            elif model_type.startswith("nrm"):
                for config in os.listdir(os.path.join(path, model_type)):

                    config_details = config.split("_")
                    state_number = int(config_details[2])
                    symbol_number = int(config_details[5])

                    path_nrm = os.path.join(path, model_type, config)

                    accuracies_nrm = get_sequence_accuracies(path_nrm)
                    smoothed_accuracies_nrm = smooth_from_list(accuracies_nrm, 400)

                    training_rewards_nrm = get_training_rewards(path_nrm)
                    smoothed_training_rewards_nrm = smooth_from_list(training_rewards_nrm, 400)

                    rows.append({
                        "Task": f"Task {task_number}",
                        "Model": "NRM",
                        "State Number": state_number,
                        "Symbol Number": symbol_number,
                        "Smoothed Accuracy": smoothed_accuracies_nrm,
                        "Smoothed Reward": smoothed_training_rewards_nrm
                    })

    df = pd.DataFrame(rows)

    # ... (Your existing data extraction code) ...
    print(df)



    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    from tqdm import tqdm

    # ... (Paste your existing functions: get_training_rewards, get_sequence_accuracies, smooth_from_list here) ...
    # ... (Paste your existing data loading loop here) ...
    # df = pd.DataFrame(rows) 

    # ==========================================
    # LATEX-READY PLOTTING CODE (Optimized Scaling)
    # ==========================================

    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Define Colors
    latex_blue   = (31/255, 119/255, 180/255)  # NRM High
    latex_orange = (255/255, 127/255, 14/255)  # NRM Low
    latex_green  = (44/255, 160/255, 44/255)   # RNN

    # 2. Labels and Palette
    def create_label(row):
        if row['Model'] == 'RNN':
            return "RNN"
        else:
            return f"NRM (S={int(row['State Number'])}, Sym={int(row['Symbol Number'])})"

    df['Configuration'] = df.apply(create_label, axis=1)

    configs = sorted(df['Configuration'].unique())
    palette = {}

    nrm_configs = [c for c in configs if "NRM" in c]
    nrm_configs.sort(key=lambda x: int(x.split("S=")[1].split(",")[0]), reverse=True)

    if "RNN" in configs:
        palette["RNN"] = latex_green
    if len(nrm_configs) > 0:
        palette[nrm_configs[0]] = latex_blue
    if len(nrm_configs) > 1:
        palette[nrm_configs[1]] = latex_orange 

    print(f"Color Mapping: {palette}")

    # 3. Plotting Loop
    unique_tasks = sorted(df['Task'].unique())

    for task in tqdm(unique_tasks, desc="Generating Optimized Plots"):
        task_df = df[df['Task'] == task]
        task_num = task.split(" ")[1] 

        # --- PLOT A: REWARDS (Auto-Scaled) ---
        plt.figure(figsize=(4, 3))
        
        rew_df = task_df.explode('Smoothed Reward')
        rew_df['Step'] = rew_df.groupby(level=0).cumcount()
        rew_df['Smoothed Reward'] = rew_df['Smoothed Reward'].astype(float)

        # Add a light gray line at 0 for reference (helps read the chart)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        sns.lineplot(
            data=rew_df, x='Step', y='Smoothed Reward',
            hue='Configuration', palette=palette, linewidth=2, legend=False
        )
        
        # FORMATTING:
        # We REMOVE plt.ylim(-100, 100) to let it auto-scale.
        # We add a grid to make reading values easier.
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.title("")
        plt.xlabel("")
        plt.ylabel("")
        plt.tick_params(labelsize=12)
        plt.tight_layout(pad=0.2)
        
        plt.savefig(os.path.join(output_dir, f"Task_{task_num}_Reward.png"), dpi=300)
        plt.close()

        # --- PLOT B: ACCURACY (Fixed 0-100) ---
        acc_df = task_df.dropna(subset=['Smoothed Accuracy'])
        if not acc_df.empty:
            plt.figure(figsize=(4, 3))
            
            plot_acc = acc_df.explode('Smoothed Accuracy')
            plot_acc['Step'] = plot_acc.groupby(level=0).cumcount()
            plot_acc['Smoothed Accuracy'] = plot_acc['Smoothed Accuracy'].astype(float)

            sns.lineplot(
                data=plot_acc, x='Step', y='Smoothed Accuracy',
                hue='Configuration', palette=palette, linewidth=2, legend=False
            )

            # Accuracy is visually better if strictly 0-105 (small buffer at top)
            plt.ylim(0, 105) 
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.title("")
            plt.xlabel("")
            plt.ylabel("")
            plt.tick_params(labelsize=12)
            plt.tight_layout(pad=0.2)
            
            plt.savefig(os.path.join(output_dir, f"Task_{task_num}_Accuracy.png"), dpi=300)
            plt.close()

        print(f"Optimized plots saved to {output_dir}")