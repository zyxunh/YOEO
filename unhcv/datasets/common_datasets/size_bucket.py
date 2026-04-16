import numpy as np


class SizeBucket:
    def __init__(self, bucket=None, size=None, scale=1, stride=64):
        if bucket is None:
            # bucket = [(704, 320), (320, 768), (896, 256), (640, 384), (576, 448), (832, 256),
            #           (512, 512), (448, 576), (384, 640), (1024, 256), (320, 704),
            #           (768, 320), (960, 256), (256, 1024), (256, 960), (256, 896), (256, 832)]
            # bucket = (
                      # [256, 1024], [256, 960], [256, 896], [256, 832],
                      # [320, 768], [320, 704], [384, 640], [448, 576],
                      # [512, 512],
                      # [512, 448], [576, 448],
                      # [512, 384], [576, 384], [640, 384],
                      # [576, 320], [640, 320], [704, 320], [768, 320],
                      # [640, 256], [704, 256], [768, 256], [832, 256], [896, 256], [960, 256], [1024, 256])
            # bucket = []
            bucket = [[256, 1024], [256, 960], [256, 896], [256, 832], [320, 768], [320, 704], [384, 640],
                      [448, 576], [512, 512], [576, 448], [640, 384], [704, 320], [768, 320], [832, 256], [896, 256],
                      [960, 256], [1024, 256]]
        bucket = np.array(bucket)
        if scale != 1:
            bucket = np.maximum(np.round((bucket * scale) / stride) * stride, stride).astype(np.int64)
        if size is not None:
            raise NotImplementedError
        # bucket = np.concatenate([bucket[1:, ::-1][::-1], bucket])
        # print(bucket[:, 0] / bucket[:, 1])
        # breakpoint()
        self.bucket = bucket
        self.bucket_ratio = bucket[:, 0] / bucket[:, 1]

    def match(self, size_wh):
        if isinstance(size_wh, (list, tuple)):
            size_wh: np.ndarray = np.array(size_wh)
        ndim = size_wh.ndim
        if ndim == 1:
            size_wh = size_wh[None]
        ratio_diff = np.abs((size_wh[:, 0] / size_wh[:, 1])[None] - self.bucket_ratio[:, None])
        match_size = self.bucket[ratio_diff.argmin(0)]
        if ndim == 1:
            match_size = match_size[0]
        return match_size

if __name__ == "__main__":
    # w = np.arange(192, 769, 64)
    # h = np.ones_like(w) * 768
    # wh = np.stack([w, h], axis=-1)
    # print(wh[:, 1] / wh[:, 0])
    # breakpoint()
    size_bucket = SizeBucket()
    128, 512
    print(size_bucket.match((100, 200)))
    print(size_bucket.match([(100, 200), (100, 100)]))