from moima.pipeline.config import Config
import pytest
import os


def test_default():
    config = Config()
    assert hasattr(config, 'dataset')
    assert hasattr(config, 'model')
    assert hasattr(config, 'splitter')
    assert hasattr(config, 'loss_fn')
    assert hasattr(config, 'optimizer')
    assert hasattr(config, 'featurizer')
    assert hasattr(config, 'scheduler')
    assert hasattr(config, 'general')

def test_io(helpers):
    config = Config()
    yaml_path = os.path.join(pytest.TEMP_PATH, 'test_config.yaml')
    json_path = os.path.join(pytest.TEMP_PATH, 'test_config.json')
    helpers.remove_files(yaml_path, json_path)
    
    config.to_yaml(yaml_path)
    config.to_json(json_path)

    yaml_config = Config.from_file(yaml_path)
    for key, value in config.__dict__.items():
        assert yaml_config.__dict__[key] == value
    
    json_config = Config.from_file(json_path)
    for key, value in config.__dict__.items():
        assert json_config.__dict__[key] == value
        
    helpers.remove_files(yaml_path, json_path)
    
    assert config.get_hash_key() == yaml_config.get_hash_key() 
    assert config.get_hash_key() == json_config.get_hash_key() 