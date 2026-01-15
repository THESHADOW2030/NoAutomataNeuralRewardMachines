import os
import sys
#python plot_smooth.py "Results"
#take the first arg
base_folder = sys.argv[1] #base name of the folder
#ENTER RECUSSIVELY in the folder and then, if there a file called sequence_classification_accuracy_[i].txt, apply the following steps


for root, dirs, files in os.walk(os.path.join(os.path.dirname(__file__), base_folder)):
    for file in files:
        if file.startswith('sequence_classification_accuracy_') and file.endswith('.txt'):
            arg = os.path.join(root, file)
            print(f"Processing file: {arg}")


            with open(arg, 'r') as f:
                lines = f.readlines()

            lines = [line.strip() for line in lines if line.strip()]
            lines = list(map(float, lines))
            smoothed = []
            mean = 0
            for i, line in enumerate(lines):
                if len(smoothed)<= 100:
                    mean += line
                    smoothed.append((i, mean / (i + 1)))
                else:
                    mean += line - lines[i - 100]
                    smoothed.append((i, mean / 100))
            lines = smoothed

            #plot the smoothed values
            import matplotlib.pyplot as plt
            plt.plot(*zip(*lines))
            plt.xlabel('Training Steps (x100)')
            plt.ylabel('Sequence Classification Accuracy')
            plt.title('Smoothed Sequence Classification for task: \n'+ root.split('/')[-3] + ' - ' + root.split('/')[-1] + ' ')
            plt.grid()
            plt.savefig(arg.replace('.txt', '_smoothed.png'))
            plt.close()

        if file.startswith('train_rewards_') and file.endswith('.txt'):
            arg = os.path.join(root, file)
            print(f"Processing file: {arg}")


            with open(arg, 'r') as f:
                lines = f.readlines()

            lines = [line.strip() for line in lines if line.strip()]
            lines = list(map(float, lines))
            smoothed = []
            mean = 0
            #sliding window of 100
            for i, line in enumerate(lines):
                if len(smoothed)<= 100:
                    mean += line
                    smoothed.append((i, mean / (i + 1)))
                else:
                    mean += line - lines[i - 100]
                    smoothed.append((i, mean / 100))
            lines = smoothed


            #plot the smoothed values
            import matplotlib.pyplot as plt
            plt.plot(*zip(*lines))
            plt.xlabel('Training Steps (x100)')
            plt.ylabel('Train Rewards')
            plt.title('Smoothed Train Rewards for task: \n'+ root.split('/')[-3] + ' - ' + root.split('/')[-1] + ' ')
            plt.grid()
            plt.savefig(arg.replace('.txt', '_smoothed.png'))
            plt.close()





