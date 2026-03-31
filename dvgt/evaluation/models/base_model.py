from abc import ABC, abstractmethod
from typing import Dict, Tuple

class BaseEvaluatorModel(ABC):
    def __init__(self, model_config: Dict = {}, eval_config: Dict = {}):
        self.model_config = model_config
        self.eval_config = eval_config
        self.model = None

    @abstractmethod
    def load(self, checkpoint_path: str):
        pass

    @abstractmethod
    def infer(self, batch) -> Tuple[Dict, Dict]:
        pass