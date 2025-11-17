#open this /mnt/ssd2/hazem/Projects/NeuralRewardMachines/Results/task1: visit(pickaxe, lava)/nrm_image_env/NUM_STATES_5_NUM_SYMBOLS_5/traces/exp0.pkl
import pickle
import os


path = '/mnt/ssd2/hazem/Projects/NeuralRewardMachines/Results/task1: visit(pickaxe, lava)/nrm_image_env/NUM_STATES_5_NUM_SYMBOLS_5/traces/exp0.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
print(data)
print(len(data))

#open this /mnt/ssd2/hazem/Projects/NeuralRewardMachines/Results/task1: visit(pickaxe, lava)/nrm_image_env/NUM_STATES_5_NUM_SYMBOLS_5/traces/exp0.pkl