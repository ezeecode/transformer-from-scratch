from pathlib import Path


def get_config():

    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "es",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tozenizer_file": "tozenizer_{0}.json",
        "experiment name": "runs/tmodel_experiment"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    
    model_filename = f"{model_basename}epoch_{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)