
from unit_ import process_data

from unit_ import read_len_

from generate_ import users_list, normal_list


# %% 选择的通道
selected_channel = [
    'F7-T7', 'T7-P7',         # 左颞叶
    'F8-T8', 'T8-P8-0',       # 右颞叶
    'T7-FT9', 'FT10-T8',      # 颞叶深部
    'FP1-F3', 'FP2-F4',       # 前额叶
    'F3-C3', 'F4-C4',         # 额中央区
    'C3-P3', 'C4-P4',         # 中央至顶叶
    'FZ-CZ', 'CZ-PZ',         # 中线结构
    'P7-O1', 'P8-O2'          # 枕叶
]