import os
import random
import time
import torch
import torch.nn as nn
from tqdm import tqdm               
import torch.optim as optim                                               
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from model_.dataset_ import getsome_, users_list
from model_.unit_ import set_seed, device, huber_loss, mask_
from model_.unit_ import MSU_, Class_


# ! Hyperparameter
data_path = r"F:\python_Work_space\MSU_TL\Data_"
set_seed(seed=508)

batch_size = 8
pre_epoch = 100
down_epoch = 2  

mask_ratio = 0.6
scheduler_patience = 2
early_stop_patience = 5
loss_pre = huber_loss

# selected_users = random.sample(users_list, 7)
selected_users =  users_list

for i in selected_users:
    print(i)
    best_pretrain_path = f"temp_best_pretrain_user_{i}.pth"
    
    # data
    pre_dataset = getsome_(root_dir=data_path, skip_subjects=[i], step_sizes=[50,50])
    train_dataset = getsome_(root_dir=data_path, skip_subjects=[i], step_sizes=[130,50])
    test_dataset = getsome_(root_dir=data_path, only_subjects=[i], step_sizes=[50,50])

    pre_dataloader = DataLoader(pre_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ! Pre-training
    pre_model = MSU_()
    pre_model.to(device)
    
    best_loss = float('inf')
    no_improve_epoch = 0
    opt_pre = optim.Adam(pre_model.parameters(), lr=0.001)
    scheduler_pre = ReduceLROnPlateau(opt_pre, mode='min', factor=0.5, patience=scheduler_patience)

    pre_model.train()
    for epoch in range(pre_epoch):
        total_loss = 0
        current_lr = opt_pre.param_groups[0]['lr']
        progress_bar = tqdm(pre_dataloader, 
                            desc=f'[Pre] Epoch {epoch + 1}/{pre_epoch}',
                            # dynamic_ncols=False,
                            ncols=120,
                            mininterval=5,
                            ascii=True,
                            leave=False
                            )
        
        for input_x, _ in progress_bar:
            opt_pre.zero_grad() 
            masked_data, mask = mask_(input_x, mask_ratio=mask_ratio)

            mask = mask.to(device) 
            input_x = input_x.to(device)
            masked_data = masked_data.to(device)
            
            encoder_x, recon_x = pre_model(masked_data)
            loss = loss_pre(input_x, recon_x, mask)
            
            loss.backward()  
            opt_pre.step()  
            total_loss += loss.item()
            if progress_bar.n > 0 and progress_bar.n % 200 == 0:
                progress_bar.set_postfix({
                    'loss':  f'{loss.item():.4f}', 
                    'lr': current_lr,
                    'avg_loss':total_loss / len(pre_dataloader),
                    })

        avg_loss = total_loss / len(pre_dataloader)
        scheduler_pre.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epoch = 0 
            torch.save(pre_model.state_dict(), best_pretrain_path)
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= early_stop_patience:
                print(f"[Pre] STOP Epoch {epoch + 1}, best Loss: {best_loss}")
                break

    # ! Downstream Fine-tuning
    pre_model = MSU_()
    pre_model.load_state_dict(torch.load(best_pretrain_path, weights_only=True))
    pre_model.to(device)
    pre_model.eval()

    class_model = Class_(512, 2) 
    class_model.to(device)
    
    criterion_clf = nn.CrossEntropyLoss()
    optimizer_clf = optim.Adam(class_model.parameters(), lr=0.001)
    
    for epoch in range(down_epoch):
        class_model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, 
                            desc=f'[Down] Epoch {epoch + 1}/{down_epoch}',
                            # dynamic_ncols=False,
                            ncols=120,
                            mininterval=5,
                            ascii=True,
                            leave=False
                            )
        
        for input_x, label in progress_bar:
            optimizer_clf.zero_grad()
            input_x = input_x.to(device)
            label = label.to(device)
            
            with torch.no_grad():
                features = pre_model(input_x)[0][-1] 
            
            Predict = class_model(features)
            loss = criterion_clf(Predict, label)

            loss.backward()
            optimizer_clf.step()
            total_loss += loss.item()

            if progress_bar.n > 0 and progress_bar.n % 50 == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })

    # ! Test
    pre_model.eval()
    class_model.eval()
    
    all_pred = []
    all_labels = []
    all_probs = [] 

    with torch.no_grad():
        progress_bar_test = tqdm(test_dataloader, 
                                desc='[Test]:',
                                # dynamic_ncols=False,
                                ncols=120,
                                mininterval=5,
                                ascii=True,
                                leave=False
                                )
        for inputs, labels in progress_bar_test:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            features = pre_model(inputs)[0][-1] 
            outputs = class_model(features) 
            
            _, pred = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_pred.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_pred) * 100                              
    sensitivity = recall_score(all_labels, all_pred, average='binary') * 100              
    specificity = recall_score(all_labels, all_pred, pos_label=0, average='binary') * 100  
    precision = precision_score(all_labels, all_pred, average='binary') * 100             
    f1 = f1_score(all_labels, all_pred, average='binary') * 100                        
    auc = roc_auc_score(all_labels, all_probs) * 100

    print(f"[Sub {i} result] :")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Sensitivity: {sensitivity:.2f}%")
    print(f"Specificity: {specificity:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"F1 Score:  {f1:.2f}%")
    print(f"AUC:       {auc:.2f}%\n")
    print("-"*70)

    time.sleep(2)
    if os.path.exists(best_pretrain_path):
        os.remove(best_pretrain_path)

