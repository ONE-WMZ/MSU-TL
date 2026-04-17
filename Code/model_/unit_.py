import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=508):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False         


#  mask fun
def mask_X(masked_data, mask_ratio=0.6, mask_value=0.0):
    if isinstance(masked_data, torch.Tensor):
        masked_data = masked_data.clone()
        rand_tensor = torch.rand_like(masked_data, dtype=torch.float32)
        mask = rand_tensor < mask_ratio
        masked_data[mask] = mask_value
        return masked_data, mask
    elif isinstance(masked_data, np.ndarray):
        masked_data = masked_data.copy()
        mask = np.random.rand(*masked_data.shape) < mask_ratio
        masked_data[mask] = mask_value
        return masked_data, mask
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")



#  class
class model_class(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        y_ = self.classifier(x)
        return y_


#  channel init : mix
class channel_init(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.channel_init = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.channel_init(x)


#  Multi-scale module
class multi_(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv_size_5 = nn.Conv1d(ch_in, ch_out, kernel_size=5, padding=2)  
        self.conv_size_10 = nn.Conv1d(ch_in, ch_out, kernel_size=11, padding=5)        
        self.conv_size_20 = nn.Conv1d(ch_in, ch_out, kernel_size=21, padding=10)               
        self.conv_hollow = nn.Conv1d(ch_in, ch_out, kernel_size=3, padding=2, dilation=2)  
        self.relu = nn.ReLU()                                                                  

    def forward(self, x):
        x_5 = self.conv_size_5(x)
        x_10 = self.conv_size_10(x)
        x_20 = self.conv_size_20(x)
        x_hollow = self.conv_hollow(x)
        centroid = (x_5 + x_10 + x_20 + x_hollow) / 4                                            
        return self.relu(centroid)


#  U-Net > down sampling
class down_(nn.Module):
    def __init__(self, ch_in, ch_out, in_size, out_size, dropout=0.3):
        super().__init__()
        self.multi_ = multi_(ch_in, ch_out)
        self.downsample = nn.MaxPool1d(kernel_size=int(in_size/out_size), stride=int(in_size/out_size))
        self.dropout = nn.Dropout(dropout)       

    def forward(self, x):
        x_multi = self.multi_(x)
        x_down = self.downsample(x_multi)
        x_down = self.dropout(x_down)             
        return x_multi, x_down


# Encoder
class encoder_(nn.Module):
    # input shape (16,400)
    def __init__(self, class_is_no):
        super().__init__()
        self.class_is_no = class_is_no                             # isNon class
        self.channel_init = channel_init(16, 16)                   # channel init
        self.down_1 = down_(16, 32, 400, 200)
        self.down_2 = down_(32, 64, 200, 100)
        self.down_3 = down_(64, 128, 100, 50)
        self.down_4 = down_(128, 256, 50, 10)
        self.down_5 = down_(256, 512, 10, 1)

    def forward(self, x):
        # X.shape = (16,400)
        x0 = self.channel_init(x)
        x1_skip, x1_down = self.down_1(x0)                         # 16->32
        x2_skip, x2_down = self.down_2(x1_down)                    # 32->64
        x3_skip, x3_down = self.down_3(x2_down)                    # 64->128
        x4_skip, x4_down = self.down_4(x3_down)                    # 128->256
        x5_skip, x5_down = self.down_5(x4_down)                    # 256->512
        feature = x5_down.flatten(start_dim=1)                     # flat
        # isnon class
        if self.class_is_no:
            return feature
        else:
            return x1_skip, x2_skip, x3_skip, x4_skip, x5_skip, x5_down, feature


# Up sampling
class up_(nn.Module):
    def __init__(self, ch_in, ch_out, skip_channels, output_size, dropout=0.3):
        super().__init__()
        self.upsample = nn.Upsample(size=output_size, mode='linear', align_corners=True)
        self.conv_input = nn.Conv1d(ch_in, ch_out, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(ch_out + skip_channels, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch_out),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x, skip):
        x_up = self.upsample(x)
        x_up = self.conv_input(x_up)

        if skip is not None:
            if x_up.shape[-1] != skip.shape[-1]:
                skip = F.interpolate(skip, size=x_up.shape[-1], mode='linear', align_corners=True)
            x = torch.cat([x_up, skip], dim=1)
        else:
            x = x_up
        return self.dropout(self.conv(x))

# Decoder
class decoder_(nn.Module):
    def __init__(self, class_is_no):
        super().__init__()
        self.class_is_no = class_is_no

        self.up_1 = up_(512, 256, 512, 10)                      # 512, skip512 -> 256
        self.up_2 = up_(256, 128, 256, 50)                      # 256, skip256 -> 128
        self.up_3 = up_(128, 64, 128, 100)                      # 128, skip128 -> 64
        self.up_4 = up_(64, 32, 64, 200)                        # 64,  skip64  -> 32
        self.up_5 = up_(32, 16, 32, 400)                        # 32,  skip32  -> 32
        self.final_conv = nn.Conv1d(16, 16, kernel_size=1)      # end -> 16

    def forward(self, all_x):
        if not self.class_is_no:
            x1_skip, x2_skip, x3_skip, x4_skip, x5_skip, x5_down, feature = all_x
            x = self.up_1(x5_down, x5_skip)
            x = self.up_2(x, x4_skip)
            x = self.up_3(x, x3_skip)
            x = self.up_4(x, x2_skip)
            x = self.up_5(x, x1_skip)
            x_ = self.final_conv(x)
            return x_
        else:
            print('Decoder not class model')
            return None


class mask_recon_X(nn.Module):
    def __init__(self, class_is_no):
        super().__init__()
        self.class_is_no = class_is_no
        self.encoder = encoder_(class_is_no)
        self.decoder = decoder_(class_is_no)

    def forward(self, x):
        if not self.class_is_no:                              # IsNon down class
            encoder_x = self.encoder(x)                       # encoder: extract feature
            decoder_x = self.decoder(encoder_x)               # decoder: Re-construct
            return encoder_x, decoder_x
        else:
            encoder_x = self.encoder(x)                       # encoder
            return encoder_x

