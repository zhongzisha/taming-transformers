import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class CelebAHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FacesHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQTrain(size=size, keys=keys)
        d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQValidation(size=size, keys=keys)
        d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class GantaTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/GantaHQ"
        with open("data/GantaHQ_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        print('train paths: ', paths)
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class GantaValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/GantaHQ"
        with open("data/GantaHQ_val.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        print('val paths: ', paths)
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class GantaHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        self.data = GantaTrain(size=size, keys=keys)
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = 0
        return ex


class GantaHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        self.data = GantaValidation(size=size, keys=keys)
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = 0
        return ex  # ex is a dictionary, keys is ["image", "coord", "class"]






class Ganta_Aug_Train(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/GantaHQ_Aug"
        with open("data/Ganta_3_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        print('train paths: ', paths)
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class Ganta_Aug_Validation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/GantaHQ_Aug/"
        with open("data/Ganta_3_val.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        print('val paths: ', paths)
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class GantaHQ_Aug_Train(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        self.data = Ganta_Aug_Train(size=size, keys=keys)
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = 0
        return ex


class GantaHQ_Aug_Validation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        self.data = Ganta_Aug_Validation(size=size, keys=keys)
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = 0
        return ex  # ex is a dictionary, keys is ["image", "coord", "class"]
