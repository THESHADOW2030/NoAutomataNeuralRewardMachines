import os
def plot_smoothed_sequence_classification_accuracy():
#read sequence_classification_accuracy_0.txt
    for i in range(5):
    #check if file exists
        if not os.path.exists(os.path.join(os.path.dirname(__file__), f'sequence_classification_accuracy_{i}.txt')):
            continue 
        with open(os.path.join(os.path.dirname(__file__), f'sequence_classification_accuracy_{i}.txt'), 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines if line.strip()]
        lines = list(map(float, lines))
        smoothed = []
        mean = 0
        for i, line in enumerate(lines):
            if i % 100 == 0:
                smoothed.append((i, mean / 100))
                mean = 0
            mean += line
            if i == len(lines) - 1:
                smoothed.append((i, mean / (i % 100 + 1)))  # handle last segment
        lines = smoothed

        #plot the smoothed values
        import matplotlib.pyplot as plt
        plt.plot(*zip(*lines))
        plt.xlabel('Training Steps (x100)')
        plt.ylabel('Sequence Classification Accuracy')
        plt.title('Smoothed Sequence Classification Accuracy over Training Steps')
        plt.grid()
        plt.savefig(os.path.join(os.path.dirname(__file__), f'sequence_classification_accuracy_smoothed_{i}.png'))
        plt.close()





        print(list(lines))