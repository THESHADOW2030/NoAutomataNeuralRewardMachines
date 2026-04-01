#open the following pickle /mnt/ssd2/hazem/Projects/NeuralRewardMachines/Results_Rush/task3: seq_visit(pickaxe, lava)/nrm_image_env/NUM_STATES_50_NUM_SYMBOLS_5/traces/exp0.pkl
import pickle
with open('/mnt/ssd2/hazem/Projects/NeuralRewardMachines/Results_Rush/task3: seq_visit(pickaxe, lava)/nrm_image_env/NUM_STATES_50_NUM_SYMBOLS_5/traces/exp0.pkl', 'rb') as f:
    data = pickle.load(f)


print(data[870][0])



def confusion_matrix_symbol_symbol():
    pass