from pathlib import Path
import numpy as np
from PIL import Image
import importlib
import sys
from typing import Any
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import math
import pdb

class PoisonedDataset(DatasetFolder):
    def __init__(self, root, loader, poison_class, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        self.poison_class = poison_class
        super().__init__(root, loader, extensions, transform, target_transform, is_valid_file)
        

    def find_classes(self, directory):
        return ([self.poison_class], {self.poison_class: int(self.poison_class)})
    
class Dataset():
    def __init__(self, data_dir, train_dir, test_dir, poison_train_dir, poison_test_dir,
                 val_dir=None, poison_val_dir=None, dataset = "GTSRB", poison_class = "00008"):

        self.dataset = dataset
        self.mean = np.array([0.,0.,0.])
        self.std = np.array([0.,0.,0.])

        self.mean_p = np.array([0., 0., 0.])
        self.std_p = np.array([0., 0., 0.])

        self.data_dir = data_dir
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.poison_train_dir = Path(poison_train_dir) if poison_train_dir is not None else None
        self.poison_test_dir = Path(poison_test_dir) if poison_test_dir is not None else None
        self.extensions =  [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
        self.poison_class = poison_class

        if val_dir is not None:
            self.val_dir = val_dir
        if poison_val_dir is not None:
            self.poison_val_dir = poison_val_dir

        self.get_label_data()
    def get_label_data(self):
        try:
            from classification_datasets.GTSRB import label_map
            # label_map = importlib.import_module("label_map", package=self.data_dir)
            self.label_map = getattr(label_map, "label_map")
            self.num_classes = len(self.label_map)
        except ModuleNotFoundError:
            print(f"No label map found in {self.data_dir}.\n")

    def calc_stats(self):
        ''' Get mean and std for all images in the test/train/poison_test/poison_train directories'''
        #pdb.set_trace()
        base_train_files = []
        base_test_files = []

        poisoned_train_files = []
        poisoned_test_files = []
        poisoned_files = []

        for ext in self.extensions:
            base_train_files += list(self.train_dir.rglob(f'*{ext}'))
            base_test_files += list(self.test_dir.rglob(f'*{ext}'))
            if self.poison_test_dir is not None:
                poisoned_train_files += list(self.poison_train_dir.rglob(f'*{ext}'))
                poisoned_test_files += list(self.poison_test_dir.rglob(f'*{ext}'))


        files = base_train_files + base_test_files
        print(f"len files: {len(files)}\n")
        if self.poison_test_dir is not None:
            poisoned_files = poisoned_train_files + poisoned_test_files

        stdTemp = np.array([0.,0.,0.])

        '''
                Calculate for clean Images...
        '''

        numSamples = len(files)
        #%%
        for i in range(numSamples):
            im = np.array(Image.open(files[i]))
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.

            for j in range(3):
                self.mean[j] += np.mean(im[:,:,j])

        self.mean = (self.mean/numSamples)

#%%
        for i in range(numSamples):
            im = np.array(Image.open(files[i]))
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.
            for j in range(3):
                stdTemp[j] += ((im[:,:,j] - self.mean[j])**2).sum()/(im.shape[0]*im.shape[1])

        self.std = np.sqrt(stdTemp/numSamples)



        '''
            Repeat for poisoned Images...
        '''

        stdTemp = np.array([0.,0.,0.])

        numSamples = len(poisoned_files)
        #%%
        for i in range(numSamples):
            im = np.array(Image.open(poisoned_files[i]))
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.

            for j in range(3):
                self.mean_p[j] += np.mean(im[:,:,j])

        self.mean_p = (self.mean_p/numSamples)
        #
        for i in range(numSamples):
            im = np.array(Image.open(poisoned_files[i]))
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.
            for j in range(3):
                stdTemp[j] += ((im[:,:,j] - self.mean_p[j])**2).sum()/(im.shape[0]*im.shape[1])

        self.std_p = np.sqrt(stdTemp/numSamples)



    def pil_loader(self, path: str) -> Image.Image:
        ''' Load a pill image'''
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    # TODO: specify the return type
    def accimage_loader(self, path: str) -> Any:
        import accimage  # type: ignore
        ''' acc images?'''
        try:
            return accimage.Image(path)
        except OSError:
            # Potentially a decoding problem, fall back to PIL.Image
            return self.pil_loader(path)

    def default_loader(self, path: str) -> Any:
        ''' Default image loader for an image from the path: path'''
        from torchvision import get_image_backend

        if get_image_backend() == "accimage":
            return self.accimage_loader(path)
        else:
            return self.pil_loader(path)


    def get_dataset_loaders(self, train_path, test_path, poison_train_path, poison_test_path, batch_size,sub_train_loader_num_im=2000, sub_train_loader_batch_size = 100 ):


        train_dataset_clean = ImageFolder(train_path, self.build_train_transform(self.mean, self.std))
        self.train_loader_clean = DataLoader(train_dataset_clean, batch_size=batch_size, shuffle=True, num_workers=28,
                                        pin_memory=True)

        test_dataset_clean = ImageFolder(test_path, self.build_valid_transform(self.mean, self.std))
        self.test_loader_clean = DataLoader(test_dataset_clean, batch_size=batch_size, num_workers=28, pin_memory=True)

        if poison_train_path is not None:
            # When finetuning, we want to use the split dataset with both clean and backdoored images
            train_dataset_poison = ImageFolder(poison_train_path, self.build_train_transform(self.mean_p, self.std_p))
            # train_dataset_poison = PoisonedDataset(poison_train_path, self.default_loader, poison_class=self.poison_class, extensions=self.extensions,
                                                #    transform=self.build_train_transform(self.mean_p, self.std_p))
            self.train_loader_poison = DataLoader(train_dataset_poison, batch_size=batch_size, num_workers=28, pin_memory=True)

            # The test dataset for poison should get only the poisoned images (not the images from attack label from split dataset)
            test_dataset_poison = PoisonedDataset(poison_test_path, self.default_loader, poison_class=self.poison_class, extensions=self.extensions,
                                            transform=self.build_valid_transform(self.mean_p, self.std_p))
            self.test_loader_poison = DataLoader(test_dataset_poison, batch_size=batch_size, num_workers=28, pin_memory=True)

        sub_train_loader_num_im = 2000
        sub_train_loader_batch_size = 100
        self.sub_train_loader = self.build_sub_train_loader(self.train_loader_clean, sub_train_loader_num_im,
                                                  sub_train_loader_batch_size, train_path, self.mean,
                                                  self.std)
        # print(len(train_loader))

    def build_train_transform(self, mean, std, im_size=224):
        # image_size = [128, 160, 192, 224]
        image_size = im_size
        color_transform = None
        resize_transform_class = transforms.Resize
        train_transforms = [
            resize_transform_class((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
        ]
        train_transforms.append(transforms.ColorJitter(brightness=32. / 255., saturation=0.5))
        train_transforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, mean, std, im_size=224):
        image_size = im_size
        return transforms.Compose([
            transforms.Resize((int(math.ceil(image_size / 0.875)), int(math.ceil(image_size / 0.875)))),
            transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def build_sub_train_loader(self, train_loader, n_images, batch_size, train_data_path, mean, std, num_worker=None,
                               num_replicas=None, rank=None):
        num_worker = train_loader.num_workers
        n_samples = len(train_loader.dataset.samples)
        g = torch.Generator()
        g.manual_seed(937162211)
        rand_indexes = torch.randperm(n_samples, generator=g).tolist()

        new_train_dataset = ImageFolder(train_data_path, self.build_train_transform(mean, std))
        chosen_indexes = rand_indexes[:n_images]
        sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
        sub_data_loader = torch.utils.data.DataLoader(
            new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
            num_workers=num_worker, pin_memory=True,
        )
        ret_list = []
        for images, labels in sub_data_loader:
            ret_list.append((images, labels))
        return ret_list

