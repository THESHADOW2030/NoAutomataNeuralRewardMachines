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
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# Add this import at the top
from torch.cuda.amp import autocast, GradScaler

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

        self.deepAutoma.to(device)
        self.classifier.to(device)

        # 1. PERSIST OPTIMIZER
        self.params = list(self.classifier.parameters()) + list(self.deepAutoma.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=5e-5)

        # 2. PERSIST SCHEDULER
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-05, verbose=True
        )

        # 3. PERSIST TEMPERATURE
        self.temperature = 2.0  # Start high, decay over time

    def set_dataset_old(self, image_traj, rew_traj):
        
        dataset_acceptances = torch.LongTensor(rew_traj)
        dataset_traces = torch.stack([torch.stack(inner) for inner in image_traj])

        image_seq_dataset = ([dataset_traces], [], [dataset_acceptances], [dataset_traces], [], [dataset_acceptances])
        self.train_img_seq, self.train_traces, self.train_acceptance_img, self.test_img_seq_hard, self.test_traces, self.test_acceptance_img_hard = image_seq_dataset

        return image_seq_dataset
    def set_dataset(self, image_traj, rew_traj):
    
    # Create Labels Tensor
        dataset_acceptances = torch.LongTensor(rew_traj)

        # Create Images Tensor
        # Optimization: Convert DoubleTensor (float64) to FloatTensor (float32) 
        # to save 50% RAM and speed up training.
        dataset_traces = torch.stack([torch.stack(inner) for inner in image_traj])

        # Pack into the tuple structure your class expects
        # Structure: (Train_X, _, Train_Y, Test_X, _, Test_Y)
        image_seq_dataset = (
            [dataset_traces],      # Train Images
            [],                    # Unused
            [dataset_acceptances], # Train Labels
            [dataset_traces],      # Test Images (We use same for now)
            [],                    # Unused
            [dataset_acceptances]  # Test Labels
        )

        # Assign to class attributes
        self.train_img_seq, self.train_traces, self.train_acceptance_img, \
        self.test_img_seq_hard, self.test_traces, self.test_acceptance_img_hard = image_seq_dataset

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



    # def train_symbol_grounding_boh(self, num_of_epochs, batch_size=16, env = None):
            
    #         # 1. PREPARE DATA LOADER
    #         train_data_tensor = self.train_img_seq[0] 
    #         train_label_tensor = self.train_acceptance_img[0].type(torch.LongTensor)
    #         train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
            
    #         # USE FULL BATCH
    #         full_batch_size = len(train_dataset)
    #         train_loader = DataLoader(train_dataset, batch_size=full_batch_size, shuffle=True)
            
    #         print(f"_____________TRAINING START_____________")
    #         print(f"Samples: {len(train_dataset)} | Actual Batch Size: {full_batch_size} (Full Batch)")

    #         self.deepAutoma.to(device)
    #         self.classifier.to(device)
            
            
            
    #         params = list(self.classifier.parameters()) + list(self.deepAutoma.parameters())
    #         optimizer = torch.optim.Adam(params, lr=0.0001)

    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-05, verbose=True
    #         )

    #         max_accuracy = 0
    #         best_classifier = self.classifier
            
    #         # Temperature (Standard annealing is fine now)
    #         self.temperature = 2.0      
    #         best_test_acc = 0.0
    #         patience = 10
    #         patience_counter = 0

    #         for epoch in range(num_of_epochs):
    #             epoch_losses = []
    #             optimizer.zero_grad()
                
    #             for batch_idx, (batch_imgs, batch_lbls) in enumerate(train_loader):
    #                 batch_imgs = batch_imgs.to(device)
    #                 batch_lbls = batch_lbls.to(device)
    #                 target_rew = batch_lbls.view(-1)
                    
    #                 curr_batch_size = batch_imgs.size(0)
    #                 length_seq = batch_imgs.size(1)

    #                 # --- A. Forward Pass (Classifier) ---
    #                 if self.dataset == 'minecraft_image':
    #                     flat_imgs = batch_imgs.view(-1, self.num_channels, self.pixels_v, self.pixels_h)
    #                     logits = self.classifier(flat_imgs) 
    #                 else:
    #                     logits = self.classifier(batch_imgs)

    #                 # --- PI INTERVENTION: PARTIAL SUPERVISION MASKS ---
    #                 # We don't need "guessing" masks. We use ground truth masks.
                    
    #                 # Mask 1: Empty Images (Reward 0)
    #                 empty_mask = (target_rew == 0)
                    
    #                 # Mask 2: Gem Images (Reward > 0)
    #                 gem_mask = (target_rew != 0)

    #                 # --- C. Gumbel Softmax ---
    #                 # We pass raw logits. The supervision below handles the guidance.
    #                 sym_sequences = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
    #                 sym_sequences = sym_sequences.view(curr_batch_size, length_seq, self.numb_of_symbols)

    #                 # --- D. Forward Pass (Automa) ---
    #                 pred_states, pred_rew = self.deepAutoma(sym_sequences, self.temperature)
    #                 pred_rew = pred_rew.view(-1, self.numb_of_rewards)

    #                 # --- E. LOSS CALCULATION ---
                    
    #                 # 1. RL Loss (Standard)
    #                 weights = torch.tensor([1.0, 50.0, 50.0]).to(device) 
    #                 rew_loss = F.nll_loss(torch.log(pred_rew + 1e-9), target_rew, weight=weights)

    #                 # 2. Supervised "Empty" Loss
    #                 # If Reward is 0, the Symbol MUST be 0.
    #                 # We treat this as a classification problem.
    #                 if empty_mask.any():
    #                     empty_logits = logits[empty_mask]
    #                     empty_targets = torch.zeros(empty_logits.size(0), dtype=torch.long).to(device)
    #                     supervised_loss = F.cross_entropy(empty_logits, empty_targets)
    #                 else:
    #                     supervised_loss = 0.0

    #                 # 3. Separation "Gem" Constraint
    #                 # If Reward is NOT 0, the Symbol MUST NOT be 0.
    #                 # We penalize the probability of Symbol 0 for Gems.
    #                 if gem_mask.any():
    #                     gem_logits = logits[gem_mask]
    #                     gem_probs = F.softmax(gem_logits, dim=-1)
    #                     # We want prob of Sym0 to be 0.
    #                     # Minimizing (prob_sym0) is equivalent to maximizing log(1 - prob_sym0)
    #                     sym0_probs = gem_probs[:, 0]
    #                     separation_loss = -torch.log(1.0 - sym0_probs + 1e-9).mean()
    #                 else:
    #                     separation_loss = 0.0

    #                 # TOTAL LOSS
    #                 # We give high weight (10.0) to supervision because it's Ground Truth.
    #                 loss = rew_loss + (10.0 * supervised_loss) + (10.0 * separation_loss)
                    
    #                 # -----------------------------------------------------

    #                 loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
                        
    #                 optimizer.step()
    #                 optimizer.zero_grad()
                    
    #             epoch_losses.append(loss.item())

    #             # --- F. End of Epoch Updates ---
    #             mean_loss_new = mean(epoch_losses)
    #             scheduler.step(mean_loss_new)
                
    #             # Standard Annealing
    #             if epoch > 0:
    #                  self.temperature = max(0.5, self.temperature * 0.95)

    #             if epoch % 5 == 0:
    #                 train_acc, _, _, test_acc = self.eval_all(automa_implementation='logic_circuit', temperature=1, discretize_labels=True)
    #                 print(f"Epoch {epoch} | Loss: {mean_loss_new:.4f} | Temp: {self.temperature:.4f} | Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f}")

    #                 if train_acc >= max_accuracy:
    #                     max_accuracy = train_acc
    #                     best_classifier = self.classifier 

    #                 if test_acc > best_test_acc:
    #                     best_test_acc = test_acc
    #                     best_classifier = self.classifier 
    #                     patience_counter = 0 
    #                 else:
    #                     patience_counter += 1
                        
    #                 if patience_counter >= patience:
    #                     print(f"Stopping Early: Accuracy hasn't improved for {patience * 5} epochs.")
    #                     break
                    
    #             self.eval_symbol_grounding(env = env)

    #         self.classifier = best_classifier

    def train_symbol_grounding(self, num_of_epochs, batch_size=16, env = None):
            
        # 1. PREPARE DATA LOADER
        train_data_tensor = self.train_img_seq[0] 
        train_label_tensor = self.train_acceptance_img[0].type(torch.LongTensor)
        
        # --- PI SAFETY CHECK: VALIDATE TARGETS ---
        # Ensure targets are indices (0, 1, 2) and not raw rewards (0, 100).
        # If your max label > 2, we clamp it to avoid crashing CrossEntropy.
        if train_label_tensor.max() >= self.numb_of_rewards:
            print(f"WARNING: Remapping targets. Max was {train_label_tensor.max()}")
            train_label_tensor[train_label_tensor > 0] = 1 # Map all positive rewards to Class 1
        
        train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
        
        # USE FULL BATCH
        full_batch_size = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        print(f"_____________TRAINING START_____________")
        print(f"Samples: {len(train_dataset)} | Actual Batch Size: {batch_size}")

        self.deepAutoma.to(device)
        self.classifier.to(device)
        
        # --- PI INTERVENTION 1: THE HEAD TRANSPLANT ---
        # Reset weights to kill "dead neuron" habits from previous failed runs.
        # IMPORTANT: Once training stabilizes (after ~3 successful calls), 
        # you can comment this line out to let the agent build long-term memory.
        # with torch.no_grad():
        #     self.classifier.fc2.reset_parameters()
        if self.temperature < 0.5:
            self.temperature = 1.0 
            print("Bumping Temperature to 1.0 for adaptation.")
            

        max_accuracy = 0
        best_classifier = self.classifier
        
        # Temperature: Standard annealing is fine now because Supervision is strong.
        self.temperature = 2.0      
        best_test_acc = 0.0
        patience = 10
        patience_counter = 0
        self.classifier.train()
        empty_class_idx = self.numb_of_symbols - 1 

        
        for epoch in range(num_of_epochs):
            epoch_losses = []
            self.optimizer.zero_grad()
            
            for batch_idx, (batch_imgs, batch_lbls) in enumerate(train_loader):
                batch_imgs = batch_imgs.to(device)
                batch_lbls = batch_lbls.to(device)
                target_rew = batch_lbls.view(-1)
                
                curr_batch_size = batch_imgs.size(0)
                length_seq = batch_imgs.size(1)

                # --- Forward Pass ---
                if self.dataset == 'minecraft_image':
                    flat_imgs = batch_imgs.view(-1, self.num_channels, self.pixels_v, self.pixels_h)
                    logits = self.classifier(flat_imgs) 
                else:
                    logits = self.classifier(batch_imgs)

                # --- Supervision Masks ---
                # Mask 1: Empty Images (Reward 0)
                empty_mask = (target_rew == 0)
                
                # Mask 2: Interesting Images (Reward > 0)
                item_mask = (target_rew != 0)

                # --- Gumbel Softmax & Automa Pass ---
                sym_sequences = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
                sym_sequences = sym_sequences.view(curr_batch_size, length_seq, self.numb_of_symbols)
                pred_states, pred_rew = self.deepAutoma(sym_sequences, self.temperature)
                pred_rew = pred_rew.view(-1, self.numb_of_rewards)

                # --- LOSS CALCULATION ---
                
                # 1. RL Loss (Standard)
                # Weights to handle class imbalance in rewards (0 is common, 100 is rare)
                # Calculate frequency
                class_counts = torch.bincount(target_rew, minlength=self.numb_of_rewards)

                # Logarithmic Smoothing for Weights
                # Instead of 1/count (which explodes for rare items), use 1 / log(count + 1.2)
                # This is standard in computer vision for unbalanced datasets.
                class_weights = 1.0 / (class_counts.float() + 0.1) 

#                Normalize weights so they sum to 1 (or len(classes))
                class_weights = class_weights / class_weights.sum() * self.numb_of_rewards
                rew_loss = F.nll_loss(torch.log(pred_rew + 1e-9), target_rew, weight=class_weights)

                # 2. Supervised "Empty" Loss (CORRECTED)
                if empty_mask.any():
                    empty_logits = logits[empty_mask]
                    
                    # FIX: Use empty_class_idx instead of 0
                    empty_targets = torch.full((empty_logits.size(0),), empty_class_idx, dtype=torch.long).to(device)
                    
                    supervised_loss = F.cross_entropy(empty_logits, empty_targets)
                else:
                    supervised_loss = 0.0

                # 3. Separation Constraint
                # If Reward > 0, the Symbol CANNOT be "Empty"
                if item_mask.any():
                    item_logits = logits[item_mask]
                    item_probs = F.softmax(item_logits, dim=-1)
                    
                    # We penalize the probability of the Empty Class for items
                    prob_of_empty = item_probs[:, empty_class_idx]
                    separation_loss = -torch.log(1.0 - prob_of_empty + 1e-9).mean()
                else:
                    separation_loss = 0.0
                
                entropy = -torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1).mean()

                # TOTAL LOSS
                # We trust the Supervision (Empty detection) the most
                loss = rew_loss + (5.0 * supervised_loss) + (2.0 * separation_loss) - (0.1 * entropy)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, max_norm=0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            epoch_losses.append(loss.item())
            mean_loss_new = mean(epoch_losses)
            self.scheduler.step(mean_loss_new)
            
            # Anneal Temperature
            if epoch > 0:
                self.temperature = max(0.5, self.temperature * 0.98) # Slower decay

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
        self.classifier.eval()    
        self.eval_symbol_grounding(env = env)
        self.classifier.train()

        # Load best weights from this training session
        new_state_dict = best_classifier.state_dict()
        old_state_dict = self.classifier.state_dict()
        
        # Soft Update (Polyak Averaging)
        # Tau = 0.2 means we keep 80% of old knowledge and accept 20% new
        tau = 0.2 
        
        for key in old_state_dict:
            # theta_new = tau * theta_learned + (1 - tau) * theta_old
            old_state_dict[key] = tau * new_state_dict[key] + (1 - tau) * old_state_dict[key]
            
        self.classifier.load_state_dict(old_state_dict)



    def train_symbol_grounding_bad_performance(self, num_of_epochs, batch_size=64, env=None):
            # 1. OPTIMIZATION: Ensure models are in Float32
            self.classifier.float() 
            self.deepAutoma.float()
            self.classifier.train()

            # Initialize Scaler for Mixed Precision (AMP)
            scaler = GradScaler() 

            # 2. PREPARE DATA LOADER
            train_data_tensor = self.train_img_seq[0].float() 
            train_label_tensor = self.train_acceptance_img[0].type(torch.LongTensor)

            # Safety: Clamp labels
            if train_label_tensor.max() >= self.numb_of_rewards:
                train_label_tensor[train_label_tensor > 0] = 1 

            train_dataset = TensorDataset(train_data_tensor, train_label_tensor)

            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                drop_last=True,
                pin_memory=False 
            )

            print(f"___TRAINING START (Fixed)___ | Samples: {len(train_dataset)} | Batch: {batch_size}")

            if self.temperature < 0.5:
                self.temperature = 1.0 
            
            self.temperature = 2.0      
            best_test_acc = 0.0
            patience = 10
            patience_counter = 0
            max_accuracy = 0
            
            best_classifier_state = {k: v.cpu() for k, v in self.classifier.state_dict().items()}
            
            empty_class_idx = self.numb_of_symbols - 1 

            for epoch in range(num_of_epochs):
                epoch_losses = []
                
                for batch_idx, (batch_imgs, batch_lbls) in enumerate(train_loader):
                    batch_imgs = batch_imgs.to(device, non_blocking=True)
                    batch_lbls = batch_lbls.to(device, non_blocking=True)
                    
                    target_rew = batch_lbls.view(-1)
                    curr_batch_size = batch_imgs.size(0)
                    length_seq = batch_imgs.size(1)

                    self.optimizer.zero_grad(set_to_none=True)

                    # --- 1. AMP Forward Pass (Classifier Only) ---
                    # We only want the heavy CNN convolution math to be in FP16.
                    with autocast(enabled=True): 
                        if self.dataset == 'minecraft_image':
                            flat_imgs = batch_imgs.view(-1, self.num_channels, self.pixels_v, self.pixels_h)
                            logits = self.classifier(flat_imgs) 
                        else:
                            logits = self.classifier(batch_imgs)

                    # --- 2. Float32 Critical Section ---
                    # We disable AMP for Gumbel, Automa, and Loss to prevent Nan/Inf
                    with autocast(enabled=False):
                        # Cast logits back to float32 for stability
                        logits = logits.float()

                        # Masks
                        empty_mask = (target_rew == 0)
                        item_mask = (target_rew != 0)

                        # Gumbel Softmax & Automa
                        sym_sequences = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
                        sym_sequences = sym_sequences.view(curr_batch_size, length_seq, self.numb_of_symbols)
                        
                        pred_states, pred_rew = self.deepAutoma(sym_sequences, self.temperature)
                        pred_rew = pred_rew.view(-1, self.numb_of_rewards)

                        # --- Loss Calculation ---
                        
                        # 1. RL Loss
                        class_counts = torch.bincount(target_rew, minlength=self.numb_of_rewards)
                        class_weights = 1.0 / (class_counts.float() + 0.1) 
                        class_weights = class_weights / class_weights.sum() * self.numb_of_rewards
                        
                        # FIX: Use Clamp instead of +1e-9. 1e-9 is too small and causes -inf in some cases.
                        # 1e-7 is safe for float32.
                        log_pred_rew = torch.log(torch.clamp(pred_rew, min=1e-7))
                        rew_loss = F.nll_loss(log_pred_rew, target_rew, weight=class_weights)

                        # 2. Supervised Loss
                        supervised_loss = 0.0
                        if empty_mask.any():
                            empty_logits = logits[empty_mask]
                            empty_targets = torch.full((empty_logits.size(0),), empty_class_idx, dtype=torch.long).to(device)
                            supervised_loss = F.cross_entropy(empty_logits, empty_targets)

                        # 3. Separation Loss
                        separation_loss = 0.0
                        if item_mask.any():
                            item_logits = logits[item_mask]
                            item_probs = F.softmax(item_logits, dim=-1)
                            prob_of_empty = item_probs[:, empty_class_idx]
                            # FIX: Clamp here as well
                            separation_loss = -torch.log(torch.clamp(1.0 - prob_of_empty, min=1e-7)).mean()
                        
                        # Entropy
                        probs = F.softmax(logits, dim=-1)
                        log_probs = F.log_softmax(logits, dim=-1)
                        entropy = -torch.sum(probs * log_probs, dim=-1).mean()

                        loss = rew_loss + (5.0 * supervised_loss) + (2.0 * separation_loss) - (0.1 * entropy)

                    # --- 3. Backward Pass ---
                    # Check for NaNs before backward to avoid crashing the scaler
                    if torch.isnan(loss):
                        print(f"Warning: Loss is NaN at Epoch {epoch}, Batch {batch_idx}. Skipping step.")
                        continue

                    scaler.scale(loss).backward()
                    
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.params, max_norm=0.5)
                    
                    scaler.step(self.optimizer)
                    scaler.update()

                    epoch_losses.append(loss.item())

                # --- End of Epoch ---
                if len(epoch_losses) > 0:
                    mean_loss_new = mean(epoch_losses)
                    self.scheduler.step(mean_loss_new)
                
                if epoch > 0:
                    self.temperature = max(0.5, self.temperature * 0.98)

                if epoch % 10 == 0:
                    with torch.no_grad():
                        train_acc, _, _, test_acc = self.eval_all(
                            automa_implementation='logic_circuit', 
                            temperature=1, 
                            discretize_labels=True
                        )
                    
                    print(f"Ep {epoch} | Loss: {mean_loss_new:.3f} | TrAcc: {train_acc:.2f} | TsAcc: {test_acc:.2f}")

                    if train_acc >= max_accuracy:
                        max_accuracy = train_acc
                        best_classifier_state = {k: v.cpu() for k, v in self.classifier.state_dict().items()}

                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_classifier_state = {k: v.cpu() for k, v in self.classifier.state_dict().items()}
                        patience_counter = 0 
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"Early Stopping @ Epoch {epoch}")
                        break

            best_classifier_state = {k: v.to(device) for k, v in best_classifier_state.items()}
            self.classifier.load_state_dict(best_classifier_state)
            
            self.classifier.eval()    
            self.eval_symbol_grounding(env=env)
            self.classifier.train()





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

        #test_accuracy_hard= eval_acceptance( self.classifier, self.deepAutoma, self.alphabet,(self.test_img_seq_hard, self.test_acceptance_img_hard), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True)

        return train_accuracy, 0,0, train_accuracy

    def eval_image_classification(self, env = None):
        train_acc = eval_image_classification_from_traces(self.custom_trace, self.symbolic_grid, self.classifier, True)
        test_acc = train_acc

                


        
        return train_acc, test_acc

    def eval_symbol_grounding_old(self, env = None):
        

        #TODO: classificare i simboli per debugging. Usare tutte le osservazione precomputed e il classificatore di Neural Reward Machine
        
        
        traces_to_test = env.env.loc_to_obs

        
        
        
        
        #use the classifier to classify the symbols
        with torch.no_grad():

            classifier_output_len = len(self.classifier(torch.randn((1,3,64,64), dtype=torch.float).to(device)).squeeze())
            predicted = [0 for _ in range(classifier_output_len)]
            for i in traces_to_test.keys():
                
                obs = traces_to_test[i]
                
                obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                logits = self.classifier(obs)
                pred_symbols = torch.argmax(logits, dim=1)
                for s in pred_symbols:
                    predicted[s] +=1
        print("Predicted symbol counts: ", predicted)

    def eval_symbol_grounding(self, env=None):
        self.classifier.eval()
        
        all_obs = []
        original_pos = env.env.agent_location
        map_size = env.env.map_size
        
        # FIX: Iterate systematically (Row-Major Order)
        # This ensures the tensor matches the print loop later
        for r in range(map_size):
            for c in range(map_size):
                env.env.agent_location = (r, c)
                obs = env.env._get_image_obs()
                all_obs.append(torch.from_numpy(obs).float())
            
        env.env.agent_location = original_pos

        # Stack and Cast
        batch_obs = torch.stack(all_obs).float().to(device) 

        with torch.no_grad():
            logits = self.classifier(batch_obs)
            pred_symbols = torch.argmax(logits, dim=1).cpu().tolist()

        # Count results
        classifier_output_len = logits.shape[1]
        predicted = [0] * classifier_output_len
        for s in pred_symbols:
            predicted[s] += 1

        print(f"Predicted symbol counts: {predicted}")

        # Visual Mapping
        # Update this list if your environment symbols change!
        chars = ['P', 'L', 'D', 'G', '.'] 

        print("\n--- Ground Truth (Real Map) ---")
        gt_str = ""
        for r in range(map_size):
            row_str = ""
            for c in range(map_size):
                # Get True Label
                true_idx = env.env.loc_to_label.get((r, c), env.env.num_symbols - 1)
                if true_idx < len(chars):
                    row_str += chars[true_idx] + " "
                else:
                    row_str += "? "
            gt_str += row_str + "\n"
        print(gt_str)

        print("--- Agent's Worldview (Prediction) ---")
        pred_str = ""
        for r in range(map_size):
            row_str = ""
            for c in range(map_size):
                # FIX: Calculate index based on Row-Major Order
                idx = r * map_size + c
                sym = pred_symbols[idx]
                
                if sym < len(chars):
                    row_str += chars[sym] + " "
                else:
                    row_str += "? "
            pred_str += row_str + "\n"
        print(pred_str)
        print("------------------------------------\n")
        
        self.classifier.train()