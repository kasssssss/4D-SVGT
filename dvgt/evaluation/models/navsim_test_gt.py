import torch.nn as nn
from .base_model import BaseEvaluatorModel
from dvgt.utils.trajectory import convert_pose_rdf_to_trajectory_flu
import copy

class PlaceHolderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

class NavsimTestGTWrapper(BaseEvaluatorModel):
    def load(self, checkpoint_path: str = ''):
        self.model = PlaceHolderModel()

    def infer(self, batch):
        # return traj directly
        batch['trajectories'] = convert_pose_rdf_to_trajectory_flu(batch['future_ego_n_to_ego_curr'][:, -1])
        predict = {
            'trajectories': copy.deepcopy(batch['trajectories'])
        }
        return batch, predict
