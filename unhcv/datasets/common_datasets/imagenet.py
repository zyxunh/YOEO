from arnold_dataset.common_datasets.torch import ImagenetDataset, ImagenetDatasetPreprocessor

imagenet_dataset = ImagenetDataset()
for idx, data in enumerate(imagenet_dataset):
    print(idx, type(data[0]), type(data[1]))
    if idx > 100:
        break
preprocessor = ImagenetDatasetPreprocessor()
for idx, data in enumerate(imagenet_dataset):
    data = preprocessor.preprocess(data)
    print(idx, type(data[0]), type(data[1]))
    if idx > 100:
        break