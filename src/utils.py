def load_params(param_file="params.yaml"):
    """Loads parameters from a YAML file."""
    import yaml
    try:
        with open(param_file, 'r') as f:
            params = yaml.safe_load(f)
        return params
    except FileNotFoundError:
        print(f"Error: Parameter file '{param_file}' not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{param_file}': {e}")
        return {}