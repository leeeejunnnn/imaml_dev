#%%
#%%
import os
import random
import numpy as np
import torch
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
#import implicit_maml.utils as utils
import utils as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.ndimage import rotate

DATA_DIR = '/home/sss-linux1/project/leejun/imaml_dev/data/omniglot/'

#%%
np.random.seed(123)
torch.manual_seed(123)
random.seed(123)

# There are 1623 characters (for Omniglot)
train_val_permutation = list(range(1623))
random.shuffle(train_val_permutation)

root = DATA_DIR
num_cls = 5
num_inst = 1
num_tasks = 20000



#%%
root1 = os.path.join(root, 'images_background')
root2 = os.path.join(root, 'images_evaluation')
num_cls = num_cls
num_inst = num_inst
#%%
# Sample num_cls characters and num_inst instances of each
languages1 = os.listdir(root1)
languages2 = os.listdir(root2)
languages1.sort()
languages2.sort()

#%%
train=True
chars = []
for l in languages1:
    chars += [os.path.join(root1, l, x) for x in os.listdir(os.path.join(root1, l))]
for l in languages2:
    chars += [os.path.join(root2, l, x) for x in os.listdir(os.path.join(root2, l))]
chars = np.array(chars)[train_val_permutation]
chars = chars[:1200] if train else chars[1200:]
random.shuffle(chars)

#%%
classes = chars[:num_cls]
labels = np.array(range(len(classes)))
labels = dict(zip(classes, labels))
instances = dict()

#%%
# Now sample from the chosen classes to create class-balanced train and val sets
train_ids = []
val_ids = []
for c in classes:
    # First get all isntances of that class
    temp = [os.path.join(c, x.decode('UTF-8')) for x in os.listdir(c)]
    instances[c] = random.sample(temp, len(temp))
    # Sample num_inst instances randomly each for train and val
    train_ids += instances[c][:num_inst]
    val_ids += instances[c][num_inst:num_inst * 2]
# Keep instances separated by class for class-balanced mini-batches
train_labels = [labels[get_class(x)] for x in train_ids]
val_labels = [labels[get_class(x)] for x in val_ids]


#%%
class OmniglotTask(object):
    """
    Create the task definition for N-way k-shot learning with Omniglot dataset
    Assumption: number of train and val instances are same (easy to lift in the future)
    """
    def __init__(self, train_val_permutation, root=DATA_DIR, num_cls=5, num_inst=1, train=True):
        """
        :param train_val_permutation: permutation of the 1623 characters, first 1200 are for train, rest for val
        :param root: location of the dataset
        :param num_cls: number of classes in task instance (N-way)
        :param num_inst: number of instances per class (k-shot)
        :param train: bool, True if meta-training phase and False if test/deployment phase
        """
        # different sampling stratergy
        # 1200 classes for meta-train phase and rest for test phase
        self.root1 = os.path.join(root, 'images_background')
        self.root2 = os.path.join(root, 'images_evaluation')
        self.num_cls = num_cls
        self.num_inst = num_inst
        # Sample num_cls characters and num_inst instances of each
        languages1 = os.listdir(self.root1)
        languages2 = os.listdir(self.root2)
        languages1.sort()
        languages2.sort()
        chars = []
        for l in languages1:
            chars += [os.path.join(self.root1, l, x) for x in os.listdir(os.path.join(self.root1, l))]
        for l in languages2:
            chars += [os.path.join(self.root2, l, x) for x in os.listdir(os.path.join(self.root2, l))]
        chars = np.array(chars)[train_val_permutation]
        chars = chars[:1200] if train else chars[1200:]
        random.shuffle(chars)
        classes = chars[:num_cls]
        labels = np.array(range(len(classes)))
        labels = dict(zip(classes, labels))
        instances = dict()
        # Now sample from the chosen classes to create class-balanced train and val sets
        self.train_ids = []
        self.val_ids = []
        for c in classes:
            # First get all isntances of that class
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            instances[c] = random.sample(temp, len(temp))
            # Sample num_inst instances randomly each for train and val
            self.train_ids += instances[c][:num_inst]
            self.val_ids += instances[c][num_inst:num_inst * 2]
        # Keep instances separated by class for class-balanced mini-batches
        self.train_labels = [labels[self.get_class(x)] for x in self.train_ids]
        self.val_labels = [labels[self.get_class(x)] for x in self.val_ids]

    def get_class(self, instance):
        return '/' + os.path.join(*instance.split('/')[:-1])
