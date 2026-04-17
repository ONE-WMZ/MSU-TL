import time
import torch
import random
import torch.nn as nn
from tqdm import tqdm                                                           # 进度条
import torch.optim as optim                                                     # 优化器
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from model_ import users_list, balance_size
from model_.unit_ import set_seed, device, mask_recon_X, model_class
from model_.dataset_ import getsome_

                                        
# train + test
def train_and_test(model_, train_loader_, test_loader_, criterion_, optimizer_, device_, num_epoch_):
    set_seed()
    model_.to(device_)
    epoch_losses = []
    epoch_metrics = []
    for epoch in range(num_epoch_):
        time.sleep(1)
        model_.train()
        total_loss = 0
        current_lr = optimizer_.param_groups[0]['lr']
        progress_bar_train = tqdm(train_loader_, desc=f'Epoch {epoch + 1}/{num_epoch_}')
        for input_x, label in progress_bar_train:
            input_x, label = input_x.to(device), label.to(device)
            with torch.no_grad():
                pre_model.eval()
                feature = pre_model(input_x)
            cls = model_(feature)
            loss = criterion_(cls, label)
            optimizer_.zero_grad()
            loss.backward()
            optimizer_.step()
            total_loss += loss.item()
            progress_bar_train.set_postfix({'loss': loss.item(), 'lr': current_lr})
        avg_loss = total_loss / len(train_loader_)
        epoch_losses.append(avg_loss)

        # test
        metrics = evaluate(model_, test_loader_, device)
        epoch_metrics.append(metrics)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        print(f"Test Metrics: "
              f"Accuracy: {metrics['accuracy']:.4f}%, "
              f"Sensitivity: {metrics['sensitivity']:.4f}%, "
              f"Specificity: {metrics['specificity']:.4f}%, "
              f"F1 Score: {metrics['f1']:.4f}%, "
              f"AUC: {metrics['auc']:.4f}%, "
              )
        print('------------------------------------------------------------------------------\n')

# test
def evaluate(model_, data_loader_, device_):
    pre_model.eval()
    model_.eval()
    all_pred = []
    all_labels = []
    with torch.no_grad():
        progress_bar_test = tqdm(data_loader_, desc='[Test]:')
        for inputs, labels in progress_bar_test:
            inputs, labels = inputs.to(device_), labels.to(device_)
            feature = pre_model(inputs)
            outputs = model_(feature)
            _, pred = torch.max(outputs.data, 1)

            all_pred.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_pred) * 100                              
    sensitivity = recall_score(all_labels, all_pred, average='binary') * 100              
    specificity = recall_score(all_labels, all_pred, pos_label=0, average='binary') * 100  
    precision = precision_score(all_labels, all_pred, average='binary') * 100             
    f1 = f1_score(all_labels, all_pred, average='binary') * 100                        
    auc = roc_auc_score(all_labels, all_pred) * 100                                        
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc': auc,
    }



# cross : 23 -> 1
def get_1():
    for i in users_list:
        class_model = model_class(512, 2)
        criterion_ = nn.CrossEntropyLoss()
        optimizer_ = optim.Adam(class_model.parameters(), lr=0.001)

        train_dataset = getsome_(root_dir="Data_", skip_subjects=i, step_sizes=balance_size)
        test_dataset = getsome_(root_dir="Data_", only_subjects=i, step_sizes=balance_size)

        train_dataloader = DataLoader(train_dataset,batch_size=8, shuffle=True,worker_init_fn=set_seed(508))
        test_dataloader = DataLoader(test_dataset,batch_size=8, shuffle=True,worker_init_fn=set_seed(508))
        train_and_test(model_=class_model,
                  train_loader_=train_dataloader,
                  test_loader_=test_dataloader,
                  criterion_=criterion_,
                  optimizer_=optimizer_,
                  device_=device,
                  num_epoch_=2,
                  )

# cross : 20 -> 4
def get_4():
    select_4 = random.sample(users_list, 4)
    print(select_4)

    class_model = model_class(512, 2)
    criterion_ = nn.CrossEntropyLoss()
    optimizer_ = optim.Adam(class_model.parameters(), lr=0.001)

    train_dataset = getsome_(root_dir="Data_", skip_subjects=select_4, step_sizes=balance_size)
    test_dataset = getsome_(root_dir="Data_", only_subjects=select_4, step_sizes=balance_size)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, worker_init_fn=set_seed(508))
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, worker_init_fn=set_seed(508))

    train_and_test(model_=class_model,
              train_loader_=train_dataloader,
              test_loader_=test_dataloader,
              criterion_=criterion_,
              optimizer_=optimizer_,
              device_=device,
              num_epoch_=2,
              )

# inner
def get_in():
    for i in users_list:
        class_model = model_class(512, 2)
        criterion_ = nn.CrossEntropyLoss()
        optimizer_ = optim.Adam(class_model.parameters(), lr=0.001)

        data_ = getsome_(root_dir="Data_", only_subjects=i, step_sizes=balance_size)

        dataloader_ = DataLoader(data_, batch_size=8, shuffle=True, worker_init_fn=set_seed(508))
        train_and_test(model_=class_model,
                  train_loader_=dataloader_,
                  test_loader_=dataloader_,
                  criterion_=criterion_,
                  optimizer_=optimizer_,
                  device_=device,
                  num_epoch_=2,
                  )


if __name__ == "__main__":
    set_seed()
    pre_model_dict = '../pre_best.pth'
    pre_model = mask_recon_X(class_is_no=True).to(device)                     
    pre_model.load_state_dict(torch.load(f=pre_model_dict, weights_only=True))   
    for param in pre_model.parameters():                                         
        param.requires_grad = False
    pre_model.eval()  
    get_1()
    # get_4()
    # get_in()     