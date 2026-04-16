import torch


__all__ = ["gaussian_noise", "poisson_noise"]


def gaussian_noise(input_tensor, std=1.):
    noise = torch.randn_like(input_tensor)
    if std != 1:
        noise = noise * std
    return noise


def poisson_noise(input_tensor):
    noise = torch.poisson(input_tensor)
    return noise


if __name__ == "__main__":
    # Test code
    import cv2
    from unhcv.common.image import visual_tensor

    image = cv2.imread("/home/yixing/dataset/coco/val2017/000000147498.jpg")
    quality = 20
    image = cv2.resize(image, (640, 640))
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
    image.requires_grad_(True)
    noise = gaussian_noise(image, std=0.1)
    image1 = visual_tensor(image + noise, max_value=1, min_value=0)
    vals = 10 ** 4
    noise = poisson_noise(image * vals) / vals
    noise = noise - image.detach() + image

    breakpoint()
    breakpoint()
    image2 = visual_tensor(noise, max_value=1, min_value=0)
    noise = poisson_noise(image.mean(1, keepdim=True) * vals) / vals
    image3 = visual_tensor(image + noise - image.mean(1, keepdim=True), max_value=1, min_value=0)
    cv2.imwrite(f"/home/yixing/train_outputs/test_gaussian.png", image1)
    cv2.imwrite(f"/home/yixing/train_outputs/test_poisson.png", image2)
    cv2.imwrite(f"/home/yixing/train_outputs/test_poisson_gray.png", image3)