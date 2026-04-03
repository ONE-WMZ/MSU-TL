import os
import mne
import numpy as np
import warnings

from data_get_ import selected_channel


# %% 数据预处理
def process_data(file_path):
    """ 数据预处理:
            1. 读取原始EDF文件
            2. 重采样到200Hz
            3. 删除重复的通道（根据名称）[可选择指定]
            4. 将数据从 V 转换为 μV，并更新单位信息
            5. 带通滤波：0.5 - 45 Hz
            6. 获取数据并进行 Z-score 标准化（逐通道）
            得到的数据形状：shape = (n_channels, n_times) = (16,_)
    """
    mne.set_log_level('WARNING')
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.resample(200.0)                                
    raw.apply_function(lambda x: x * 1e6)              
    for ch in raw.info['chs']:
        ch['unit'] = 105                                 
    raw.filter(l_freq=0.5, h_freq=45, method='fir',           
                fir_window='hamming', n_jobs=2)
    selected_channels = selected_channel                  
    available_channels = raw.info['ch_names']            
    missing_channels = [ch for ch in selected_channels if ch not in available_channels]
    if missing_channels:
        print(f"文件缺少以下通道：{missing_channels}，跳过此文件。")
        return None
    else:
        raw_copy = raw.copy().pick(selected_channels)       
    data = raw_copy.get_data()                               
    data_mean = np.mean(data, axis=1, keepdims=True)            
    data_std = np.std(data, axis=1, keepdims=True)              
    data_normalized = (data - data_mean) / (data_std + 1e-10)   
    return data_normalized                                      


# %% 读取数据长度
def read_len_(dir_path='../Data_'):
    for i in range(1, 25):
        user_id = f"chb{i:02d}"
        user_dir = os.path.join(dir_path, user_id)
        for filename in ['label_0.npy', 'label_1.npy']:
            file_path = os.path.join(user_dir, filename)
            if os.path.exists(file_path):
                try:
                    data = np.load(file_path)
                    if len(data.shape) < 2:
                        print(f"文件 {filename} 数据维度不足，跳过。")
                        continue
                    length = data.shape[1]
                    time_seconds = length / (200*60)  
                    print(f"[{user_id}] 文件 {filename} 的数据时长为: {time_seconds:.2f} 分 (采样点数: {length})")
                except Exception as e:
                    print(f"[{user_id}] 读取文件 {filename} 出错: {e}")
            else:
                print(f"[{user_id}] 文件 {filename} 不存在。")


# %% end

