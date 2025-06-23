import pandas as pd


def validate_training_configs_for_inference(configs):
    # Makes sure that the keys in config_inference_params_from_train match across configs, if using an
    # ensemble model. Otherwise it may lead to unpredictable results.

    # Here is the list of important parameters that should match between configuration(s) files
    config_inference_params_from_train = {'BASEMODEL': ['Target_Patch_Size', 'Target_Pixel_Size', 'Precision'],
                                          'PROCESSING_TYPE': ['Model'],
                                          'ADVANCEDMODEL': ['Pretrained'],
                                          'CRITERIA': ['classes', 'IF_markers']}


    # Initialize a dictionary to store the new configuration
    new_config = {}
    original_values = {}

    # Flag to indicate if a conflict was found
    conflict_found = False

    # Loop over the keys in the predefined parameters
    for main_key, sub_keys in config_inference_params_from_train.items():
        for sub_key in sub_keys:

            # Initialize a set to store unique values for the current parameter
            values = set()

            # Iterate through each configuration
            for config in configs:
                # Check if both the main and sub key are in the configuration
                if main_key in config and sub_key in config[main_key]:
                    value = config[main_key][sub_key]
                    original_values[(main_key, sub_key)] = value # store original value
                    if isinstance(value, list):
                        value = tuple(value)
                    values.add(value)
                else:
                    print(f"Missing {sub_key} in {main_key}")
                    conflict_found = True
                    break

            # Check if there is more than one unique value for the current parameter
            if len(values) > 1:
                print(f"Conflict found for {main_key} -> {sub_key}: {values}")
                conflict_found = True
                break
            elif values:
                # If values set is not empty, add the key-value pair to the new configuration
                new_config.setdefault(main_key, {})[sub_key] = original_values[(main_key, sub_key)]


    # If no conflict was found, the new_config can be used or saved
    if not conflict_found:
        print("New configuration created successfully:", new_config)
        # Optionally, save new_config to a file or use it further
    else:
        print("Configuration conflict detected. Please resolve before proceeding.")


    return new_config


def merge_configs(inference_config, training_config):
    # Iterate through each key and subkey in the training configuration
    for main_key, sub_dict in training_config.items():
        for sub_key, value in sub_dict.items():
            # Check if the key/subkey exists in the inference configuration
            if main_key in inference_config and sub_key in inference_config[main_key]:
                # If the value is different, issue a warning and update it
                if inference_config[main_key][sub_key] != value:
                    print(f"Warning: '{main_key} -> {sub_key}' is different in the inference config file. " +
                          "It will be updated to the correct value from the training config.")
            # Update or add the key/subkey to the inference configuration
            inference_config.setdefault(main_key, {})[sub_key] = value

    return inference_config

