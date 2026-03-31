import torch
import time
import logging
from hydra.utils import instantiate

from .base_model import BaseEvaluatorModel

class DVGT2Wrapper(BaseEvaluatorModel):
    def load(self, checkpoint_path: str):
        self.model = instantiate(self.model_config, _recursive_=False)
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self.model.load_state_dict(state_dict, strict=True)

    def infer(self, batch):
        infer_window = self.eval_config.get('infer_window', -1)
        assert infer_window < 24, (
            f"Inference window ({infer_window}) exceeds the training limit. "
            "The current implementation is optimized for sequences < 24 frames."
        )
        
        # images: [1, T, V, C, H, W]
        model_start_time = time.time()
        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            if infer_window > 0:
                # streaming evaluation
                predictions = self.model.inference(images=batch["images"], ego_status=batch.get('ego_status'), infer_window=infer_window)
            else:
                predictions = self.model(images=batch["images"], ego_status=batch.get('ego_status'))
        model_end_time = time.time()
        logging.debug(f"model_infer: {(model_end_time - model_start_time):.4f}s")
    
        postprocess_results = self.model.post_process_for_eval(predictions, batch, **self.eval_config)

        return batch, postprocess_results

