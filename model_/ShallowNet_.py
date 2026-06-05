import torch
import torch.nn as nn


# ! model: ShallowNet
class ShallowNet(nn.Module):
    """
    “Deep learning with convolutional neural networks for EEG decoding and visualization”
    https://robintibor.github.io/eeg-deep-learning-phd-thesis/DeepArchitectures.html#shallow-convnet-architecture
    """
    def __init__(self, num_channels=16, num_timepoints=400, num_classes=2):
        super().__init__()
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=40,
                kernel_size=(1, 25),
                stride=(1, 1),
                padding=(0, 12),
                bias=False
            ),
            nn.BatchNorm2d(40),
        )
        # Spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=40,
                out_channels=40,
                kernel_size=(num_channels, 1),
                groups=40,
                bias=False
            ),
            nn.BatchNorm2d(40),
            nn.ELU(),
        )
        # Mean pooling
        self.pool = nn.AvgPool2d(
            kernel_size=(1, 15),
            stride=(1, 15)
        )
        self.dropout = nn.Dropout(0.5)
        # Automatically infer feature dimension
        with torch.no_grad():
            x = torch.zeros(1, 1, num_channels, num_timepoints)
            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.pool(x)
            self.feature_dim = x.numel()
        # Classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # (B,C,T) -> (B,1,C,T)
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        logits = self.classifier(x)
        return logits


# ! Train(23-1)
import numpy as np
from tqdm import tqdm   
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from dataset_ import getsome_

if __name__ == "__main__":
    data_path = r"F:\python_Work_space\MSU_TL\Data_"
    # Hyperparameters
    num_epochs = 100
    batch_size = 8
    learning_rate = 0.001
    num_classes = 2
    num_timesteps = 400
    num_channels = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_log = {
    "Subject": [],
    "Accuracy": [],
    "Sensitivity": [],
    "Specificity": [],
    "Precision": [],
    "F1 Score": [],
    }
    users_list = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06',
                  'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12',
                  'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18',
                  'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24']
    
    print("ShallowNet")
    for i in users_list:
        print(i)
        train_dataset = getsome_(root_dir=data_path, skip_subjects=[i], step_sizes=[130,50])
        test_dataset = getsome_(root_dir=data_path, only_subjects=[i], step_sizes=[50,50])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        model = ShallowNet(
            num_channels=16,
            num_timepoints=400,
            num_classes=2
        )
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            total_loss, correct = 0, 0
            total_samples = 0
            progress_bar = tqdm(train_dataloader, desc=f'[Train] Epoch {epoch + 1}/{num_epochs}')
            for data, labels in progress_bar:
                total_samples += labels.size(0)
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)  # Shape: (batch_size, num_classes)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                progress_bar.set_postfix({'loss': loss.item(), 
                                          'avg_loss':total_loss / total_samples,
                                          'Acc':correct / total_samples,
                                            })

        all_pred = []
        all_labels = []
        model.eval()
        total_loss, correct = 0, 0
        total_samples = 0
        
        progress_bar = tqdm(test_dataloader, desc=f'[Test] {epoch + 1}/{num_epochs}')

        with torch.no_grad():
            for data, labels in progress_bar:
                total_samples += labels.size(0)
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                correct += (outputs.argmax(dim=1) == labels).sum().item()

                all_pred.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'Acc':correct / total_samples,})
        accuracy = accuracy_score(all_labels, all_pred) * 100                              
        sensitivity = recall_score(all_labels, all_pred, average='binary') * 100              
        specificity = recall_score(all_labels, all_pred, pos_label=0, average='binary') * 100  
        precision = precision_score(all_labels, all_pred, average='binary') * 100             
        f1 = f1_score(all_labels, all_pred, average='binary') * 100   

        print(f"[Sub {i} result] :")
        print(f"Accuracy:  {accuracy:.2f}%")
        print(f"Sensitivity: {sensitivity:.2f}%")
        print(f"Specificity: {specificity:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"F1 Score:  {f1:.2f}%")
        print("-"*70)

        results_log["Subject"].append(i)
        results_log["Accuracy"].append(accuracy)
        results_log["Sensitivity"].append(sensitivity)
        results_log["Specificity"].append(specificity)
        results_log["Precision"].append(precision)
        results_log["F1 Score"].append(f1)

    # ! mean and std
    print("\n" + "-"*70)
    print(f"{'Metric':<15} | {'Mean ± Std (%)':<20}")
    print("-"*70)

    metrics_to_calc = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score"]
    for metric in metrics_to_calc:
        data = results_log[metric]
        mean_val = np.mean(data)
        std_val = np.std(data)
        print(f"{metric:<15} | {mean_val:.2f} ± {std_val:.2f}")

