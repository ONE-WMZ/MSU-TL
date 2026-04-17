import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Union, List, Set


class dataset_pre(Dataset):
    def __init__(self, root_dir, window_size=400, step_size=50):
        self.root_dir = root_dir
        self.window_size = window_size
        self.step_size = step_size
        self.data = []
        # all file dir
        for subject in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path):
                continue
            for file_name in os.listdir(subject_path):
                if not file_name.endswith('.npy'):
                    continue
                file_path = os.path.join(subject_path, file_name)
                # label_0.npy -> 0
                label = int(file_name.split('_')[1].split('.')[0])
                # load data
                data_array = np.load(file_path)  # shape: (16, T)
                # slip window
                num_channels, total_time = data_array.shape
                for start in range(0, total_time - window_size + 1, step_size):
                    window = data_array[:, start:start + window_size]  # shape: (16, 400)
                    self.data.append((window, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_, label = self.data[idx]
        # to Tensor
        x_tensor = torch.from_numpy(x_).float()  # shape: (16, 400)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, label_tensor


#  get(or skip) some sub
class getsome_(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        window_size: int = 400,
        step_sizes: Optional[Dict[int, int]] = None,
        skip_subjects: Optional[Union[List[str], List[int]]] = None,
        only_subjects: Optional[Union[List[str], List[int]]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.step_sizes = step_sizes or {}
        self.data = []

        self.skip_subjects: Set[str] = set()
        self.only_subjects: Set[str] = set()

        all_subjects = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        if skip_subjects:
            for s in skip_subjects:
                if isinstance(s, int):

                    self.skip_subjects.add(all_subjects[s])
                else:
                    self.skip_subjects.add(str(s))

        if only_subjects:
            for s in only_subjects:
                if isinstance(s, int):
                    self.only_subjects.add(all_subjects[s])
                else:
                    self.only_subjects.add(str(s))
        
        for subject in all_subjects:

            if subject in self.skip_subjects:
                continue

            if self.only_subjects and subject not in self.only_subjects:
                continue
            subject_path = self.root_dir / subject

            for file_path in subject_path.glob("*.npy"):
                try:
                    label_str = file_path.stem.split('_')[1]
                    label = int(label_str)
                except (IndexError, ValueError) as e:
                    print(f"skip invalid format file: {file_path} | error: {e}")
                    continue

                try:
                    data_array = np.load(file_path)  # (C, T)
                except Exception as e:
                    print(f"load dir fail: {file_path} | error: {e}")
                    continue

                if data_array.ndim != 2:
                    print(f"file {file_path} data dim abnorm: {data_array.shape},skip.")
                    continue
                _, total_time = data_array.shape

                if total_time < self.window_size:
                    print(f"file {file_path} len no enough ({total_time} < {self.window_size}),skip.")
                    continue

                step_size = self.step_sizes.get(label, 50)

                for start in range(0, total_time - self.window_size + 1, step_size):
                    window = data_array[:, start:start + self.window_size]  #  (C, window_size)
                    self.data.append((window, label))
        # print(f"total {len(self.data)} window data")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        window, label = self.data[idx]
        x_tensor = torch.from_numpy(window).float()     
        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor

