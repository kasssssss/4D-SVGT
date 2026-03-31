DATASET_REGISTRY = dict()

def register_dataset(name):
    def decorator(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered.")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset(name, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' is not registered.")
    dataset_class = DATASET_REGISTRY[name]
    print(f"Successfully loaded dataset: {name}")
    return dataset_class(**kwargs)