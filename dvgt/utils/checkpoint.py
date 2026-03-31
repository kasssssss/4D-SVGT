# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import (
    Any,
    Dict,
    List,
)

import torch
import torch.nn as nn
import os
from iopath.common.file_io import g_pathmgr
from wcmatch import fnmatch

# ------------------------------------------------------------
# Glob‑matching flags (behave like the Unix shell) 
# ------------------------------------------------------------
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)

class DDPCheckpointSaver:
    def __init__(
        self,
        checkpoint_folder: str,
        checkpoint_names: List[str],
        rank: int,
        epoch: int,
    ):
        super().__init__()
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        assert len(self.checkpoint_names) <= 2, "Only support up to two checkpoint names"
        self.worker_id = rank
        self.epoch = epoch

    def save_checkpoint(
        self,
        model: nn.Module,
        **kwargs: Any,
    ) -> None:
        checkpoint = dict(**kwargs)
        checkpoint["model"] = model.state_dict()

        if self.worker_id == 0:
            if len(self.checkpoint_names) == 1:
                # 只有一个名字时，它就是源文件
                source_name = self.checkpoint_names[0]
                link_name = None
            else:
                # 有两个名字时，我们约定第一个是通用名（链接），第二个是带epoch的（源）
                source_name = self.checkpoint_names[1]
                link_name = self.checkpoint_names[0]

            source_path = os.path.join(
                self.checkpoint_folder, f"{source_name}.pt"
            )
            logging.info(
                f"Saving checkpoint at epoch {self.epoch} to {source_path}"
            )
            robust_torch_save(checkpoint, source_path)

            if link_name is not None:
                link_path = os.path.join(
                    self.checkpoint_folder, f"{link_name}.pt"
                )
                # 如果链接已存在，先删除旧的，再创建新的
                # 确保 "checkpoint.pt" 总是指向最新的 epoch 文件
                if g_pathmgr.exists(link_path):
                    g_pathmgr.rm(link_path)
                
                logging.info(f"Creating hard link: {link_path} -> {source_path}")
                os.link(source_path, link_path)


def robust_torch_save(checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
    """
    A more robust version of torch.save that works better with preemptions
    and corruptions if a job is preempted during save.
    """
    # Move the existing checkpoint to a backup location
    backup_checkpoint_path = checkpoint_path + ".bak"
    backup_checkpoint_path_saved = False
    if g_pathmgr.exists(checkpoint_path):
        assert not g_pathmgr.exists(
            backup_checkpoint_path
        ), f"this should not exist... {backup_checkpoint_path}"
        g_pathmgr.mv(checkpoint_path, backup_checkpoint_path)
        backup_checkpoint_path_saved = True
    # Save the checkpoint
    with g_pathmgr.open(checkpoint_path, "wb") as f:
        torch.save(checkpoint, f)
    # Remove the backup checkpoint
    if backup_checkpoint_path_saved:
        g_pathmgr.rm(backup_checkpoint_path)