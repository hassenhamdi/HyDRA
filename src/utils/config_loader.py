# src/utils/config_loader.py
import yaml
import os

class ConfigLoader:
    _config = None

    @classmethod
    def load(cls, profile_name: str = 'development'):
        """
        Loads a configuration profile. This should be called once at application startup.
        """
        if cls._config is not None and cls._config.get('profile_name') == profile_name:
            # Avoid reloading the same profile, but allow switching profiles.
            return

        with open("configs/deployment_profiles.yaml", "r") as f:
            all_profiles = yaml.safe_load(f)
        
        if profile_name not in all_profiles["profiles"]:
            raise ValueError(f"Profile '{profile_name}' not found in deployment_profiles.yaml")
        
        print(f"--- Loading HyDRA with Deployment Profile: '{profile_name}' ---")
        cls._config = all_profiles["profiles"][profile_name]
        cls._config['profile_name'] = profile_name
        return cls._config

    @classmethod
    def get_config(cls):
        """
        Retrieves the already-loaded configuration.
        Raises a RuntimeError if the config hasn't been loaded yet.
        """

        if cls._config is None:
            raise RuntimeError("Configuration has not been loaded. Call ConfigLoader.load() first.")
        return cls._config

# Make get_config a convenient top-level function
def get_config(profile_name: str = 'development'):
    return ConfigLoader.get_config()