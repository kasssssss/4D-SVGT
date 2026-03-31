import torch
import time
import logging
from typing import Dict, Tuple
from hydra.utils import instantiate

from .base_model import BaseEvaluatorModel

class DVGT1Wrapper(BaseEvaluatorModel):
    def load(self, checkpoint_path: str):
        self.model = instantiate(self.model_config, _recursive_=False)
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self.model.load_state_dict(state_dict, strict=True)

    def infer(self, batch) -> Tuple[Dict, Dict]:
        model_start_time = time.time()

        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            predictions = self.model(batch["images"])

        model_end_time = time.time()
        logging.debug(f"model_infer: {(model_end_time - model_start_time):.4f}s")

        return batch, self.model.post_process_for_eval(predictions, batch, **self.eval_config)
