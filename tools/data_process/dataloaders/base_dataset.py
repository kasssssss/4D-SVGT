from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np

class BaseDataset(Dataset, ABC):

    def __init__(
            self, 
            data_root: str, 
            storage_pred_depth_path: str, storage_proj_depth_path: str, 
            storage_image_path: str, storage_align_depth_path: str, 
            num_tokens: int
        ) -> None:
        """
        num_tokens: moge处理图片为moge需要输入的token数
        gen_meta: 是否生成json
        gen_depth: 是否生成proj depth or pred depth
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.storage_pred_depth_path = Path(storage_pred_depth_path)
        self.storage_proj_depth_path = Path(storage_proj_depth_path)
        self.storage_image_path = Path(storage_image_path)
        self.storage_align_depth_path = Path(storage_align_depth_path)
        self.num_tokens = num_tokens

        # 预定义一个转换矩阵，几乎所有数据集都会用到
        self.T_rdf_flu = np.array([
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [1,  0,  0, 0],
            [0,  0,  0, 1]  
        ], dtype=np.float64)

        self.T_flu_rdf = np.array([
            [ 0,  0, 1, 0],
            [-1,  0, 0, 0],
            [ 0, -1, 0, 0],
            [ 0,  0, 0, 1]  
        ], dtype=np.float64)

        self.samples = self._gather_samples()
    
    @abstractmethod
    def _gather_samples(self) -> List:
        """ 
        Scan data_root and return a list of item to process. 
        同时构建output path，并且过滤已经生成的sample
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Dict:
        pass

    def __len__(self) -> int:
        return len(self.samples)