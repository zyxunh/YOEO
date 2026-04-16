import time

def get_class_inform(class_member, member_names=[]):
    repr_str = class_member.__class__.__name__
    for member_name in member_names:
        repr_str += f'({member_name}={getattr(class_member, member_name)}, \n'
    if len(member_names):
        repr_str = repr_str[:-3]
    return repr_str


def write_exception(exception, path='/home/tiger/workspace/datasets_nas/debug/exception.log'):
    with open(path, "a") as f:
        log_out = '{}: UNH_ERROR {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), exception.__repr__())
        print(log_out)
        f.write(log_out)


if __name__ == "__main__":
    try:
        k = "gaewta" + 1
    except Exception as ex:
        write_exception(ex)