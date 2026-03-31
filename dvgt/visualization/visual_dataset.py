from dvgt.datasets.scene_dataset import DVGTSceneDataset

class VisualParquetDataset(DVGTSceneDataset):
    """
        Selective Loading: Loads only the scene(s) matching the specified ID(s).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        target_keys = [
            'log-0104-scene-0007',
            'log-0002-scene-0015',
            'log-0005-scene-0009',
            'log-0006-scene-0012',
        ]
        new_sequence_list = []
        for key in self.sequence_list:
            for target in target_keys:
                if target in key:
                    new_sequence_list.append(key)
        self.sequence_list = new_sequence_list
        
        self._dataset_len = len(self.sequence_list)