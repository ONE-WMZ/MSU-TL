import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_.unit_ import mask_X, set_seed, device, mask_recon_X
from model_.dataset_ import dataset_pre
from model_.loss_ import huber_loss





# train
def train_pre(model_, dataloader_, criterion_, optimizer_, device_, num_epoch_):
    save_pt_path = '../pre_best.pth'
    model_.to(device)
    model_.train()
    best_loss = float('inf')
    epoch_losses = []
    no_improve_epoch = 0
    patience = 5 

    scheduler = ReduceLROnPlateau(optimizer_, mode='min', factor=0.5, patience=patience)
    for epoch in range(num_epoch_):
        total_loss = 0
        # current lr
        current_lr = optimizer_.param_groups[0]['lr']
        progress_bar = tqdm(dataloader_, desc=f'Epoch {epoch + 1}/{num_epoch_}')
        for input_x, _ in progress_bar:
            masked_data, mask = mask_X(input_x)
            masked_x = masked_data.to(device_)
            mask = torch.from_numpy(mask).to(device_)
            input_x = input_x.to(device_)
            encoder_x, recon_x = model(input_x)
            loss = criterion_(input_x, recon_x, mask)
            optimizer_.zero_grad()  
            loss.backward()  
            optimizer_.step()  
            total_loss += loss.item()

            progress_bar.set_postfix({'loss': loss.item(), 'lr': current_lr})

        avg_loss = total_loss / len(dataloader_)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss}")

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epoch = 0
            torch.save(model.state_dict(), save_pt_path)
            print(f"Best model saved with loss: {avg_loss}")
        else:
            no_improve_epoch += 1
            print(f"No improvement for {no_improve_epoch} epochs")
            if no_improve_epoch >= patience:
                print(f"Early stopping at epoch {epoch + 1} as no improvement for {patience} consecutive epochs")
                break
    print("[Training complete.]")
    print("epoch_loss:", epoch_losses)


if __name__ == "__main__":
    pre_loader = DataLoader(dataset=dataset_pre(root_dir="Data_"), batch_size=8, shuffle=True)
    set_seed()
    model = mask_recon_X(class_is_no=False)
    train_pre(model_=model,
            dataloader_=pre_loader,
            criterion_=huber_loss,
            optimizer_=optim.Adam(model.parameters(), lr=0.001),
            device_=device,
            num_epoch_=100
            )

