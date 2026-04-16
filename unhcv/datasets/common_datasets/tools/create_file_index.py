import os

import imagesize
from unhcv.common import get_related_path, replace_suffix
from unhcv.common.utils import walk_all_files_with_suffix, get_base_name, obj_dump, find_path


def main(args):
    indexes = []
    root = find_path(args.root)
    image_roots = args.image_roots
    save_name = args.save_name
    image_tags = args.image_tags
    image_roots_real = [os.path.join(root, var) for var in image_roots]
    file_names = walk_all_files_with_suffix(image_roots_real[0])
    for file_name in file_names:
        indexes.append({})
        file_name_lt = [file_name]
        for image_root_real in image_roots_real[1:]:
            file_name_i = get_related_path(file_name, image_roots_real[0], image_root_real)
            if not os.path.exists(file_name_i):
                file_name_i = replace_suffix(file_name_i, '.png')
            if not os.path.exists(file_name_i):
                file_name_i = replace_suffix(file_name_i, '.jpg')
            if not os.path.exists(file_name_i):
                raise FileNotFoundError(f"File not found: {file_name_i}")
            file_name_lt.append(file_name_i)
        for tag, file_name in zip(image_tags, file_name_lt):
            indexes[-1][tag] = get_base_name(file_name, input_root=root)
        indexes[-1]["size_wh"] = imagesize.get(file_name_lt[0])

    obj_dump(os.path.join(root, save_name), indexes, exist_ok=True)


def parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=None) # /home/yixing/dataset/Adobe_EntitySeg/custom_test/val_lr_inpainting_d5_1105_indexes
    parser.add_argument('--image_roots', type=str, nargs='+', default=["image", "inpainting_mask"])
    parser.add_argument('--save_name', type=str, default="index.yml")
    parser.add_argument('--image_tags', type=str, nargs='+', default=["image", "inpainting_mask"])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    main(args)