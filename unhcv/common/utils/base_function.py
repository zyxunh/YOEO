def add_id_to_key(x: dict, id_key: str):
    x_out = {}
    for key, value in x.items():
        x_out[f'{id_key}_{key}'] = value
    return x_out

def call_circulation(x):
    if isinstance(x, float):
        x = round(x, 2)    
        round()

def format_forprint_circulation(x, ndigits):
    if isinstance(x, float):
        out = float(format(x, f'.{ndigits}g'))
    elif isinstance(x, (tuple, list)):
        out = []
        for var in x:
            out.append(format_forprint_circulation(var, ndigits=ndigits))
    elif isinstance(x, dict):
        out = {}
        for key, value in x.items():
            out[key] = format_forprint_circulation(value, ndigits=ndigits)
    else:
        out = x
    return out

if __name__ == '__main__':
    out = format_forprint_circulation({'x': [0.5325325, 1.36352, 15325e-7]}, ndigits=3)
    from unhcv.common.utils import obj_dump
    obj_dump('tmp.json', out, mode_json='a')
    pass