def get_nested_key(batch, key):
    """
    Get a nested key from batch, supporting paths like "out/pixel_pred" or "rgb_norm".
    Raises KeyError if the key is not found.
    """
    keys = key.split("/")
    tensor_data = batch
    for k in keys:
        if not isinstance(tensor_data, dict) or k not in tensor_data:
            raise KeyError(
                f"Key '{key}' not found in batch. Missing part: '{k}' in {list(tensor_data.keys()) if isinstance(tensor_data, dict) else type(tensor_data)}"
            )
        tensor_data = tensor_data[k]
    return tensor_data
