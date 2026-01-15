import os
from numpy import rint
import torchvision
from PIL import Image
import torch
import pickle
from .DeepAutoma import ProbabilisticAutoma
from .NN_models import CNN_grounder, Linear_grounder
import torch.nn.functional as F
from statistics import mean
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader

from .utils import eval_acceptance, eval_learnt_DFA_acceptance, eval_image_classification_from_traces
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
import time

def create_batches_same_length(dataset, labels, size):
    new_dataset = []
    new_labels = []
    num_batches = int(len(dataset)/size)
    for i in range(num_batches):
        batch_trace = []
        batch_label = []
        for j in range(size):
            batch_trace.append(dataset[i*size+j])
            batch_label.append(labels[i*size+j])
        batch_trace = torch.stack(batch_trace)
        batch_label = torch.stack(batch_label)
        new_dataset.append(batch_trace)
        new_labels.append(batch_label)
    return new_dataset, new_labels 

class NeuralRewardMachine:
    def __init__(self, numb_states, numb_symbols, numb_rewards, num_exp=0,log_dir="Results/", dataset="minecraft_location"):
        self.first_training = False
        self.ltl_formula_string = "goal"
        self.log_dir = log_dir
        self.exp_num=num_exp

        self.numb_of_symbols = numb_symbols
        self.numb_of_states = numb_states
        self.numb_of_rewards = numb_rewards

        self.alphabet = ["c"+str(i) for i in range(self.numb_of_symbols) ]

        #################### networks
        self.hidden_dim =numb_states

        ##### DeepDFA
        self.deepAutoma = ProbabilisticAutoma(self.numb_of_symbols, self.numb_of_states, self.numb_of_rewards)

        ##### Classifier
        self.dataset = dataset
        if dataset == 'minecraft_image':
            #self.num_classes = 5
            self.num_classes = self.numb_of_symbols
            self.num_channels = 3

            self.pixels_h = 64
            self.pixels_v = 64

            self.num_features = 4 #<---??
            self.classifier = CNN_grounder(self.num_classes)

        if dataset == 'minecraft_location':
            self.num_inputs = 2
            #self.num_classes = 5
            self.num_classes = self.numb_of_symbols
            self.classifier = Linear_grounder(self.num_inputs, 8, self.num_classes)

        self.temperature = 1
        #questa resize si può togliere mi sà
        resize = torchvision.transforms.Resize((64,64))
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            resize,
        ])

        #queste cose sotto misà che servivano per fare la image classification
        '''
        if dataset == 'minecraft_image':
            trace = []
            dir = os.listdir('custom_trace_whole')
            for i in range(len(dir)):
                img = Image.open('custom_trace_whole/img'+str(i)+'.jpg')
                img = transforms(img)
                trace.append(img)
            self.custom_trace = [torch.stack(trace).unsqueeze(0)]

            trace = []
        '''
        if dataset == 'minecraft_location':
            self.custom_trace = [torch.tensor([[0,0],[0,1],[0,2],[0,3],
                                             [1,0],[1,1],[1,2],[1,3],
                                             [2,0],[2,1],[2,2],[2,3],
                                             [3,0],[3,1],[3,2],[3,3]]).unsqueeze(0)]

        self.symbolic_grid = [torch.tensor([[0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,1,0,0],
                                    [0,0,0,0,1], [1,0,0,0,0], [0,0,0,0,1], [0,0,0,0,1],
                                    [0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1],
                                    [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,1], [0,1,0,0,0]])]
        #[0,0,0,0,1] white cell
        #[0,0,0,1,0] gem
        #[0,0,1,0,0] door
        #[0,1,0,0,0] lava
        #[1,0,0,0,0] pick

    def set_dataset(self, image_traj, rew_traj):
        
        dataset_acceptances = torch.LongTensor(rew_traj)
        dataset_traces = torch.stack([torch.stack(inner) for inner in image_traj])

        image_seq_dataset = ([dataset_traces], [], [dataset_acceptances], [dataset_traces], [], [dataset_acceptances])
        self.train_img_seq, self.train_traces, self.train_acceptance_img, self.test_img_seq_hard, self.test_traces, self.test_acceptance_img_hard = image_seq_dataset

        return image_seq_dataset


    def eval_learnt_DFA(self, automa_implementation, temp, mode="dev"):
        if mode=="dev":
            if automa_implementation == 'dfa':
                train_acc = eval_learnt_DFA_acceptance(self.dfa, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
                test_acc = eval_learnt_DFA_acceptance(self.dfa, (self.dev_traces, self.dev_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
            else:
                train_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.train_traces, self.train_acceptance_tr), automa_implementation, temp)
                test_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.dev_traces, self.dev_acceptance_tr), automa_implementation, temp)
        else:
            if automa_implementation == 'dfa':
                train_acc = eval_learnt_DFA_acceptance(self.dfa, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
                test_acc = eval_learnt_DFA_acceptance(self.dfa, (self.test_traces, self.test_acceptance_tr),
                                                      automa_implementation, temp, alphabet=self.alphabet)
            else:
                train_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp)
                test_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.test_traces, self.test_acceptance_tr),
                                                      automa_implementation, temp)
        return train_acc, test_acc




    def train_symbol_grounding(self, num_of_epochs, batch_size=16, env = None):
            
            # 1. PREPARE DATA LOADER
            train_data_tensor = self.train_img_seq[0] 
            train_label_tensor = self.train_acceptance_img[0].type(torch.LongTensor)
            train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
            
            # USE FULL BATCH (Stabilizes Gradients)
            full_batch_size = len(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=full_batch_size, shuffle=True)
            
            print(f"_____________TRAINING START_____________")
            print(f"Samples: {len(train_dataset)} | Actual Batch Size: {full_batch_size} (Full Batch)")

            self.deepAutoma.to(device)
            self.classifier.to(device)
            
            params = list(self.classifier.parameters()) + list(self.deepAutoma.parameters())
            optimizer = torch.optim.Adam(params, lr=0.0001)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-05, verbose=True
            )

            max_accuracy = 0
            best_classifier = self.classifier
            
            # Annealing setup
            self.temperature = 1.0      
            best_test_acc = 0.0
            patience = 10
            patience_counter = 0

            for epoch in range(num_of_epochs):
                epoch_losses = []
                optimizer.zero_grad()
                
                # 2. MINI-BATCH LOOP (Actually Full Batch)
                for batch_idx, (batch_imgs, batch_lbls) in enumerate(train_loader):
                    
                    # Move to device
                    batch_imgs = batch_imgs.to(device)
                    batch_lbls = batch_lbls.to(device)

                    # --- CRITICAL FIX: DEFINE TARGETS FIRST ---
                    # We need these for the Masking Logic immediately
                    target_rew = batch_lbls.view(-1)
                    
                    curr_batch_size = batch_imgs.size(0)
                    length_seq = batch_imgs.size(1)

                    # --- A. Forward Pass (Classifier) ---
                    if self.dataset == 'minecraft_image':
                        flat_imgs = batch_imgs.view(-1, self.num_channels, self.pixels_v, self.pixels_h)
                        logits = self.classifier(flat_imgs) 
                    else:
                        logits = self.classifier(batch_imgs)

                    # --- B. DEADLOCK BREAKER: LOGIT MASKING ---
                    # 1. Identify the "Background Symbol" using Empty samples (Reward 0)
                    zero_rew_mask = (target_rew == 0)
                    
                    if zero_rew_mask.any():
                        with torch.no_grad():
                            zero_logits = logits[zero_rew_mask]
                            # Find which symbol has the highest avg probability on empty images
                            zero_probs = F.softmax(zero_logits, dim=-1).mean(dim=0)
                            background_symbol_idx = torch.argmax(zero_probs)
                    else:
                        background_symbol_idx = 0 # Default fallback

                    # 2. Mask this symbol for "Gem" images (Positive Rewards)
                    masked_logits = logits.clone()
                    pos_rew_mask = (target_rew > 0) # Identifies Gems/Lava

                    if pos_rew_mask.any():
                        # Set the Background Symbol logit to -infinity for Gems.
                        # This FORCES the model to pick a new symbol (breaking the deadlock).
                        masked_logits[pos_rew_mask, background_symbol_idx] = -1e9

                    # --- C. Gumbel Softmax (Use Masked Logits!) ---
                    sym_sequences = F.gumbel_softmax(masked_logits, tau=self.temperature, hard=True, dim=-1)
                    sym_sequences = sym_sequences.view(curr_batch_size, length_seq, self.numb_of_symbols)

                    # --- D. Forward Pass (Automa) ---
                    pred_states, pred_rew = self.deepAutoma(sym_sequences, self.temperature)
                    pred_rew = pred_rew.view(-1, self.numb_of_rewards)

                    # --- E. Loss Calculation ---
                    
                    # 1. Weighted Reward Loss (The "Scream" Factor)
                    # Ensure weights are on device!
                    # --- E. Loss Calculation ---
                    
                    # 1. Weighted Reward Loss (The "Scream" Factor)
                    weights = torch.tensor([1.0, 50.0, 50.0]).to(device) 
                    rew_loss = F.nll_loss(torch.log(pred_rew + 1e-9), target_rew, weight=weights)

                    # --- PI INTERVENTION: ANTI-COLLAPSE REGULARIZATION ---
                    
                    # Get soft probabilities for entropy calculation
                    # (Use the unmasked logits to encourage the base network to learn diversity)
                    probs = F.softmax(logits, dim=-1)
                    
                    # 2. Per-Sample Entropy (Encourage Uncertainty/Exploration)
                    # If the model is 100% sure of Symbol 2, this is 0. We want it > 0 initially.
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
                    
                    # 3. Batch Diversity Loss (The "Nuclear Option" for Collapse)
                    # Calculate the average probability distribution across the ENTIRE batch.
                    # If the batch is [2, 2, 2, 2], avg_prob is [0, 0, 1, 0].
                    # We want avg_prob to look like [0.2, 0.2, 0.2, 0.2, 0.2].
                    avg_probs = torch.mean(probs, dim=0)
                    batch_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-9))
                    
                    # We want to MAXIMIZE batch_entropy (make the batch diverse).
                    # Since we minimize loss, we subtract it.
                    
                    # Hyperparameters for stability
                    lambda_entropy = 0.05   # Scale for individual exploration
                    lambda_diversity = 0.5  # Scale for forcing different symbols in a batch
                    
                    # Anneal these? Ideally yes, but for now, let's just force the learning.
                    #if self.exp_num > 0: # Reduce regularization later if needed
                     #   pass 

                    # TOTAL LOSS
                    loss = rew_loss - (lambda_entropy * entropy) - (lambda_diversity * batch_entropy)
                    
                    # -----------------------------------------------------

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
                        
                    optimizer.step()
                    optimizer.zero_grad()
                    
                epoch_losses.append(loss.item())

                # --- F. End of Epoch Updates ---
                mean_loss_new = mean(epoch_losses)
                scheduler.step(mean_loss_new)
                
                # Fast Annealing
                if epoch > 0:
                    self.temperature = max(0.1, self.temperature * 0.95)

                if epoch % 5 == 0:
                    train_acc, _, _, test_acc = self.eval_all(automa_implementation='logic_circuit', temperature=1, discretize_labels=True)
                    print(f"Epoch {epoch} | Loss: {mean_loss_new:.4f} | Temp: {self.temperature:.4f} | Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f}")

                    if train_acc >= max_accuracy:
                        max_accuracy = train_acc
                        best_classifier = self.classifier 

                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_classifier = self.classifier 
                        patience_counter = 0 
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"Stopping Early: Accuracy hasn't improved for {patience * 5} epochs.")
                        break
                    
                self.eval_symbol_grounding(env = env)

            self.classifier = best_classifier
            


    def train_DFA(self, batch_size, num_of_epochs, decay=0.999, freezed=False):
        def get_lr(optim):
            for param_group in optim.param_groups:
                return param_group['lr']

        tot_size = len(self.train_traces)
        mean_loss = 1000000

        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_NS_exp"+str(self.exp_num), 'w')
        dev_file = open(self.log_dir+self.ltl_formula_string+"_dev_acc_NS_exp"+str(self.exp_num), 'w')

        train_file_dfa = open(self.log_dir+self.ltl_formula_string+"_train_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        dev_file_dfa = open(self.log_dir+self.ltl_formula_string+"_dev_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        test_file_dfa = open(self.log_dir+self.ltl_formula_string+"_test_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        loss_file = open(self.log_dir+self.ltl_formula_string+"_loss_dfa_NS_exp"+str(self.exp_num), 'w')

        cross_entr = torch.nn.CrossEntropyLoss()
        print("_____________training the DFA_____________")
        print("training on {} sequences using {} automaton states".format(tot_size, self.numb_of_states))

        params = [self.deepAutoma.trans_prob] + [self.deepAutoma.rew_matrix]
        optimizer = torch.optim.Adam(params, lr=0.01)
        sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-04)


        min_temp = 0.00001
        self.temperature =1.0

        if freezed:
            self.temperature = min_temp

        start_time = time.time()
        epoch= -1
        while True:
            epoch+=1
            print("epoch: ", epoch)
            losses = []
            for i in range(len(self.train_traces)):

                batch_trace_dataset = self.train_traces[i].to(device)
                batch_acceptance = self.train_acceptance_tr[i].to(device)
                optimizer.zero_grad()

                predictions= self.deepAutoma(batch_trace_dataset, self.temperature)

                loss = cross_entr(predictions, batch_acceptance)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='logic_circuit', temp=self.temperature)
            mean_loss_new = mean(losses)
            print("SEQUENCE CLASSIFICATION (LOGIC CIRCUIT): train accuracy : {}\ttest accuracy : {}\tloss : {}".format(train_accuracy, test_accuracy, mean_loss_new))

            train_file.write("{}\n".format(train_accuracy))
            dev_file.write("{}\n".format(test_accuracy))
            train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='logic_circuit', temp=min_temp)
            print("SEQUENCE CLASSIFICATION (DFA): train accuracy : {}\ttest accuracy : {}".format(train_accuracy, test_accuracy))

            train_file_dfa.write("{}\n".format(train_accuracy))
            dev_file_dfa.write("{}\n".format(test_accuracy))
            loss_file.write("{}\n".format(mean(losses)))
            if freezed:
                self.temperature = min_temp
            else:
                self.temperature = max(self.temperature*decay, min_temp)
            print("temp: ", self.temperature)

            sheduler.step(mean_loss_new)
            print("lr: ", get_lr(optimizer))
            if mean_loss_new < 0.318 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            if epoch > 200 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            mean_loss = mean_loss_new

        print("STO DENTO AL TRAINING DEL DFA")
        # Construct the directory path
        dir_path = self.log_dir+self.ltl_formula_string
        os.makedirs(dir_path, exist_ok=True)  # Create it if it doesn't exist

        # Now save the pickle
        with open(f"{dir_path}/deepAutoma_{self.ltl_formula_string}_exp{self.exp_num}.pkl", 'wb') as outp:
            print(f"Saving the automa in {dir_path}/deepAutoma_{self.ltl_formula_string}_exp{self.exp_num}.pkl")
            pickle.dump(self.deepAutoma, outp, pickle.HIGHEST_PROTOCOL)
        
        ######################## net2dfa
        #save the minimized dfa     CONTINUARE COL NOME 2030
        self.dfa = self.deepAutoma.net2dfa(min_temp, name_automata= self.log_dir+self.ltl_formula_string+"_exp"+str(self.exp_num)+"")


        ex_time =  time.time() - start_time

        with open("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+".ex_time", "w") as f:
            f.write("{}\n".format(ex_time))

        #print it
        try:
            self.dfa.to_graphviz().render("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_minimized.dot")
        except:
            print("Not able to render automa")
        with open("DFA_predicted_nesy/"+self.ltl_formula_string, 'wb') as outp:
            pickle.dump(self.dfa, outp, pickle.HIGHEST_PROTOCOL)

        with open("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_min_num_states", "w") as f:
            f.write(str(len(self.dfa._states)))

        #ULTIMO TEST usando il DFA sul TEST set
        train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='dfa', temp=min_temp, mode="test")
        print("FINAL SEQUENCE CLASSIFICATION ON TEST SET: {}".format(test_accuracy))

        test_file_dfa.write("{}\n".format(test_accuracy))

    def eval_all(self, automa_implementation, temperature, discretize_labels=False):
        train_accuracy = eval_acceptance(self.classifier, self.deepAutoma, self.alphabet, (self.train_img_seq, self.train_acceptance_img), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True)

        test_accuracy_hard= eval_acceptance( self.classifier, self.deepAutoma, self.alphabet,(self.test_img_seq_hard, self.test_acceptance_img_hard), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True)

        return train_accuracy, 0,0, test_accuracy_hard

    def eval_image_classification(self, env = None):
        train_acc = eval_image_classification_from_traces(self.custom_trace, self.symbolic_grid, self.classifier, True)
        test_acc = train_acc

                


        
        return train_acc, test_acc

    def eval_symbol_grounding(self, env = None):
        

        #TODO: classificare i simboli per debugging. Usare tutte le osservazione precomputed e il classificatore di Neural Reward Machine
        
        
        traces_to_test = env.env.loc_to_obs

        
        
        
        
        #use the classifier to classify the symbols
        with torch.no_grad():

            classifier_output_len = len(self.classifier(torch.randn((1,3,64,64), dtype=torch.double).to(device)).squeeze())
            predicted = [0 for _ in range(classifier_output_len)]
            for i in traces_to_test.keys():
                
                obs = traces_to_test[i]
                
                obs = torch.from_numpy(obs).double().to(device).unsqueeze(0)
                logits = self.classifier(obs)
                pred_symbols = torch.argmax(logits, dim=1)
                for s in pred_symbols:
                    predicted[s] +=1
        print("Predicted symbol counts: ", predicted)