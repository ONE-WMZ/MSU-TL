import os
import numpy as np
import pandas as pd
from unit_ import process_data

# %% 文件地址
root_path = "E:/CHB-MIT/chb-mit-scalp-eeg-database-1.0.0/"
seizure_info_path = "seizure_info.csv"

normal_list = ['chb01_01.edf', 'chb02_01.edf', 'chb03_05.edf', 'chb04_01.edf',
               'chb05_01.edf', 'chb06_02.edf', 'chb07_01.edf', 'chb08_03.edf',
               'chb09_01.edf', 'chb10_01.edf', 'chb11_01.edf', 'chb12_19.edf',
               'chb13_02.edf', 'chb14_42.edf', 'chb15_02.edf', 'chb16_01.edf',
               'chb17a_05.edf', 'chb18_02.edf', 'chb19_02.edf', 'chb20_01.edf',
               'chb21_01.edf', 'chb22_01.edf', 'chb23_10.edf', 'chb24_13.edf']

users_list = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06',
              'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12',
              'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18',
              'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24']


# %% 正常数据
def generate_normal_data(list_, save_dir='../Data_', save_label_0='label_0'):
    for user_id in list_:
        file_path = root_path + list_[user_id]
        print(file_path)
        data = process_data(file_path)
        if data is not None:

            seizure_data = data[:, 0:20 * 60 * 200]

            save_path = os.path.join(save_dir, user_id, f'{save_label_0}.npy')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  
            np.save(save_path, seizure_data)
            print('--', user_id, ':处理完成.\t', seizure_data.shape)


# %% 癫痫数据
def generate_seizure_(seizure_info_, save_dir='../Data_', save_label_1='label_1'):
    seizure_all = pd.read_csv(seizure_info_)
    for user_id in users_list:
        user_seizure_data = []
        user_list = seizure_all[seizure_all['File Name'].str.startswith(user_id)]
        print(user_list)
        for j in range(len(user_list)):
            file_name = root_path + str(user_id) + '/' + user_list.iloc[j]['File Name']
            start = user_list.iloc[j]['Seizure Start Time (seconds)']
            end = user_list.iloc[j]['Seizure End Time (seconds)']
            
            data = process_data(file_name)
            if data is None:
                continue
            else:

                seizure_data = data[:, start * 200:end * 200]
                user_seizure_data.append(seizure_data)

        merged_data = np.concatenate(user_seizure_data, axis=1)

        save_path = os.path.join(save_dir, user_id, f'{save_label_1}.npy')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  
        np.save(save_path, merged_data)
        print('--', user_id, ':处理完成.\t', merged_data.shape)

# %% start

generate_normal_data(list_ = normal_list)
generate_seizure_(seizure_info_= seizure_info_path)


# %% end
