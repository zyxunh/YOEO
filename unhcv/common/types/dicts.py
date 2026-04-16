def recursive_update_dict(in_dict, update_func):
    for key, value in in_dict.items():
        value = update_func(value)
    return
