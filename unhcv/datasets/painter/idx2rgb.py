def generate_idx2color_map(num_classes):
    num_sep_per_channel = int(num_classes ** (1 / 3)) + 1  # 19
    separation_per_channel = 256 // num_sep_per_channel

    color_list = []
    for location in range(num_classes):
        num_seq_r = location // num_sep_per_channel ** 2
        num_seq_g = (location % num_sep_per_channel ** 2) // num_sep_per_channel
        num_seq_b = location % num_sep_per_channel
        assert (num_seq_r <= num_sep_per_channel) and (num_seq_g <= num_sep_per_channel) \
               and (num_seq_b <= num_sep_per_channel)

        R = 255 - num_seq_r * separation_per_channel
        G = 255 - num_seq_g * separation_per_channel
        B = 255 - num_seq_b * separation_per_channel
        assert (R < 256) and (G < 256) and (B < 256)
        assert (R >= 0) and (G >= 0) and (B >= 0)
        assert (R, G, B) not in color_list

        color_list.append((R, G, B))
        # print(location, (num_seq_r, num_seq_g, num_seq_b), (R, G, B))

    return np.array(color_list, dtype=np.uint8)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from tqdm import tqdm
    from unhcv.common.utils import walk_all_files_with_suffix, write_im, get_related_path, remove_dir
    color_list = generate_idx2color_map(151)
    image_root = "/home/tiger/dataset/ADEChallengeData2016/annotations/training"
    save_root = "/home/tiger/dataset/ADEChallengeData2016/annotations_rgb/training"
    remove_dir(save_root)
    image_names = walk_all_files_with_suffix(image_root)
    for image_name in tqdm(image_names):
        image = cv2.imread(image_name, 0)
        image_color = np.zeros([*image.shape, 3], dtype=np.uint8)
        unique_idxes = np.unique(image)
        # unique_idxes = unique_idxes - 1
        unique_idxes = unique_idxes[(unique_idxes>=0)&(unique_idxes<=255)]
        # breakpoint()
        for unique_idx in unique_idxes:
            image_color[image == unique_idx] = color_list[unique_idx]
        write_im(get_related_path(image_name, image_root, save_root),  image_color)
