# src/utils/config_loader.py
import yaml
import os

class ConfigLoader:
    _config = None

    @classmethod
    def load(cls, profile_name: str = None):
        if cls._config is not None and profile_name is None:
            return cls._config

        with open("configs/deployment_profiles.yaml", "r") as f:
            all_profiles = yaml.safe_load(f)

        if profile_name is None:
            profile_name = os.getenv("HYDRA_PROFILE", all_profiles.get("default_profile", "development"))
        
        if profile_name not in all_profiles["profiles"]:
            raise ValueError(f"Profile '{profile_name}' not found in deployment_profiles.yaml")
        
        print(f"--- Loading HyDRA with Deployment Profile: '{profile_name}' ---")
        cls._config = all_profiles["profiles"][profile_name]
        cls._config['profile_name'] = profile_name
        return cls._config

def get_config():
    config = ConfigLoader.load()
    if config is None:
        raise RuntimeError("Configuration has not been loaded. Call ConfigLoader.load() first.")
    return config