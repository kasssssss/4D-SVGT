import torch
import time
import logging
from .base_model import BaseEvaluatorModel
from hydra.utils import instantiate
from safetensors.torch import load_file

class VGGTWrapper(BaseEvaluatorModel):
    def load(self, checkpoint_path: str = ''):
        self.model = instantiate(self.model_config, _recursive_=False)
        if checkpoint_path.endswith('.pt'):
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if 'model' in state_dict:
                state_dict = state_dict['model']
        else:
            state_dict = load_file(checkpoint_path, device="cpu")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        logging.info(f"VGGTWrapper: Missing keys: {missing}")
        logging.info(f"VGGTWrapper: Unexpected keys: {unexpected}")
        
    def infer(self, batch):        
        model_start_time = time.time()
        
        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            predictions = self.model(batch["images"])
            
        model_end_time = time.time()
        logging.debug(f"model_infer: {(model_end_time - model_start_time):.4f}s")

        return batch, self.model.post_process_for_eval(predictions, batch, **self.eval_config)
