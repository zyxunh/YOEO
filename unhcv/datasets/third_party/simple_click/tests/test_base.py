import numpy as np

from unhcv.datasets.third_party.simple_click import MultiPointSampler, DSample, get_labels_with_sizes


def main():
    points_sampler=MultiPointSampler(max_num_points=12)
    instances_mask = np.zeros([10, 10], dtype=np.uint8)
    instances_mask[2:4, 2:4] = 1; instances_mask[6:9, 6:9] = 5
    instances_ids, _ = get_labels_with_sizes(instances_mask)
    sample= DSample(None, instances_mask, objects_ids=instances_ids, sample_id=0)
    points_sampler.sample_object(sample)
    points = np.array(points_sampler.sample_points())
    mask = points_sampler.selected_mask
    breakpoint()
    pass


if __name__ == "__main__":
    main()