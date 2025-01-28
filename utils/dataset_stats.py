from pathlib import Path
import numpy as np
from PIL import Image
import importlib
import sys



class stats():
    def __init__(self, data_dir, train_dir, test_dir, poison_train_dir, poison_test_dir,
                 val_dir=None, poison_val_dir=None, dataset = "GTSRB"):

        self.dataset = dataset
        self.mean = np.array([0.,0.,0.])
        self.std = np.array([0.,0.,0.])

        self.mean_p = np.array([0., 0., 0.])
        self.std_p = np.array([0., 0., 0.])

        self.data_dir = data_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.poison_train_dir = poison_train_dir
        self.poison_test_dir = poison_test_dir

        if val_dir is not None:
            self.val_dir = val_dir
        if poison_val_dir is not None:
            self.poison_val_dir = poison_val_dir

        self.get_label_data()
    def get_label_data(self):
        try:
            label_map = importlib.import_module(self.data_dir + "/label_map.py")
            self.label_map = getattr(label_map, "label_map")
            self.num_classes = len(self.label_map)
        except ModuleNotFoundError:
            print(f"No label map found in {self.data_dir}.\n")

    def calc_stats(self):

        extensions = ['*.ppm', '*.png', '*.jpg']
        base_train_files = []
        base_test_files = []

        poisoned_train_files = []
        poisoned_test_files = []

        for ext in extensions:
            base_train_files += list(self.train_dir.rglob(f'{ext}'))
            base_test_files += list(self.test_dir.rglob(f'{ext}'))

            poisoned_train_files += list(self.poison_train_dir.rglob(f'{ext}'))
            poisoned_test_files += list(self.poison_test_dir.rglob(f'{ext}'))


        files = base_train_files + base_test_files
        poisoned_files = poisoned_train_files + poisoned_test_files

        stdTemp = np.array([0.,0.,0.])

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

#%%
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
        #%%
        for i in range(numSamples):
            im = np.array(Image.open(poisoned_files[i]))
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.
            for j in range(3):
                stdTemp[j] += ((im[:,:,j] - self.mean_p[j])**2).sum()/(im.shape[0]*im.shape[1])

        self.std_p = np.sqrt(stdTemp/numSamples)