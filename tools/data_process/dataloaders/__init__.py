from .factory import get_dataset

# 只需要导入模块触发注册流程，不需要导入具体类
from . import nuscene_loader
from . import kitti_loader
from . import open_scene_loader
from . import waymo_loader
from . import ddad_loader
from . import argoverse_loader
from . import nuplan_loader
from . import navsim_loader