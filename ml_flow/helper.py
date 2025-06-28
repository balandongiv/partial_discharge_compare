import json
import yaml
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path=r"../ml_flow/config.yaml"):
    """Load configuration from a YAML or JSON file."""
    if not os.path.exists(config_path):
        logging.warning(
            f"Configuration file '{config_path}' not found. Using default settings."
        )
        return {}

    try:
        with open(config_path, "r") as f:
            if config_path.endswith(".json"):
                config = json.load(f)
            else:
                config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from '{config_path}'")
        return config or {}
    except (json.JSONDecodeError, yaml.YAMLError):
        logging.error(
            f"Invalid configuration format in '{config_path}'. Using default settings."
        )
    except Exception as e:
        logging.error(
            f"Unexpected error loading configuration from '{config_path}': {e}."
        )
    return {}

def main():
    config = load_config()
    print("Loaded Configuration:", config)

if __name__ == '__main__':
    main()
