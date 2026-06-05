import os
import numpy as np
import pandas as pd

from unit_ import process_data
from . import users_list, channel_list, normal_list, Raw_data_path, seizure_info_csv


# ! normal data
def generate_normal_(norm_list, save_dir='../Data_', save_label_0='label_0'):
    for user_id in norm_list:
        file_path = Raw_data_path + norm_list[user_id]
        print(file_path)
        data = process_data(file_path, selected_channel=channel_list)
        if data is not None:
            # seg (20 minute)
            seizure_data = data[:, 0:20 * 60 * 200]
            # save 
            save_path = os.path.join(save_dir, user_id, f'{save_label_0}.npy')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, seizure_data)
            print('--', user_id, ':processing finish.\t', seizure_data.shape)


# ! seizure data
def generate_seizure_(seizure_info, save_dir='../Data_', save_label_1='label_1'):
    seizure_all = pd.read_csv(seizure_info)
    for user_id in users_list:
        user_seizure_data = []
        user_list = seizure_all[seizure_all['File Name'].str.startswith(user_id)]
        print(user_list)
        for j in range(len(user_list)):
            file_name = Raw_data_path + str(user_id) + '/' + user_list.iloc[j]['File Name']
            start = user_list.iloc[j]['Seizure Start Time (seconds)']
            end = user_list.iloc[j]['Seizure End Time (seconds)']
            # preprocess
            data = process_data(file_name)
            if data is None:
                continue
            else:
                # seg (all)
                seizure_data = data[:, start * 200:end * 200]
                user_seizure_data.append(seizure_data)
        # combine 
        merged_data = np.concatenate(user_seizure_data, axis=1)
        # save
        save_path = os.path.join(save_dir, user_id, f'{save_label_1}.npy')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  
        np.save(save_path, merged_data)
        print('--', user_id, ':processing finish.\t', merged_data.shape)



if False:
    generate_normal_(norm_list = normal_list)
    generate_seizure_(seizure_info = seizure_info_csv)

