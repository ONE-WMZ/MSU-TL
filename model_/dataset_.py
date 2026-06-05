from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, List, Set, Tuple


users_list = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06',
              'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12',
              'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18',
              'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24']
balance_size = [130, 50]

class getsome_(Dataset):
    def __init__(
        self,
        root_dir: str,
        window_size: int = 400,
        step_sizes: List[int] = [50, 50],
        skip_subjects: List[Union[str, int]] = [],
        only_subjects: List[Union[str, int]] = [],
    ):
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.step_sizes = step_sizes
        self.data: List[Tuple[np.ndarray, int]] = []

        self.skip_subjects: Set[str] = set()
        self.only_subjects: Set[str] = set()

        if not self.root_dir.exists():
            raise FileNotFoundError(f"No file : {self.root_dir}")
            
        all_subjects = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        for s in skip_subjects:
            if isinstance(s, int):
                if 0 <= s < len(all_subjects):
                    self.skip_subjects.add(all_subjects[s])
            else:
                self.skip_subjects.add(str(s))
        for s in only_subjects:
            if isinstance(s, int):
                if 0 <= s < len(all_subjects):
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
                except (IndexError, ValueError):
                    continue
                try:
                    data_array = np.load(file_path)
                except Exception:
                    continue
                if data_array.ndim != 2:
                    continue
                _, total_time = data_array.shape
                if total_time < self.window_size:
                    continue

                current_step = self.step_sizes[label] if label < len(self.step_sizes) else 50
                for start in range(0, total_time - self.window_size + 1, current_step):
                    window = data_array[:, start:start + self.window_size]
                    self.data.append((window.copy(), label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        window, label = self.data[idx]
        x_tensor = torch.from_numpy(window).float()     
        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor
    
    