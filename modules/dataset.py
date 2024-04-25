from glob import glob
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import torch
from os.path import join

class BimodalDataset(Dataset):

    def __init__(self, root, transform, modality='both'):
        super().__init__()

        self.transform = transform
        self.modality = modality

        self.cfp_path = os.path.join(root, "cfp")
        self.oct_path = os.path.join(root, "oct")

        self.classes = []
        self.instances = []
        self.targets = []

        for root, dirs, imgs in os.walk(self.cfp_path, topdown=True):
            for classe in sorted(dirs):
                self.classes.append(classe)
            for img in sorted(imgs):
                self.instances.append(((os.path.join(root.split('/')[-1], img))))
                label = self.classes.index(root.split('/')[-1])
                if label < 2 :
                    label = 0
                else:
                    label = 1
                self.targets.append(label)

    @staticmethod
    def np_loader(path, modality='cfp'):
        if modality == 'oct':
            path = path.split('.')[0]
            # img = [resize_with_pad(cv2.imread(bscan, cv2.IMREAD_ANYDEPTH), (512,512)) for bscan in glob(f"{path}/*") if "npy" not in bscan]
            img = [cv2.resize(cv2.imread(bscan, cv2.IMREAD_ANYDEPTH), (300,300), interpolation=cv2.INTER_AREA) for bscan in glob(f"{path}/*") if "npy" not in bscan]
            img = np.transpose(np.array(img),(1,2,0))
            # nparray = np.load(join(path,'array_19_768.npy'), allow_pickle=True)
            return img
        
        img = cv2.imread(path)
        return img
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):

        label = self.targets[idx]

        if self.modality == 'both':
            cfp_img = self.np_loader(os.path.join(self.cfp_path, self.instances[idx]))
            oct_img = self.np_loader(os.path.join(self.oct_path, self.instances[idx]), modality='oct')
            return self.transform((cfp_img, oct_img)), label
        
        elif self.modality == "cfp":
            img = self.np_loader(os.path.join(self.cfp_path, self.instances[idx]))

        elif self.modality == "oct":
            img, nparray = self.np_loader(os.path.join(self.oct_path, self.instances[idx]), modality='oct')
            # return (self.transform(img), torch.from_numpy(nparray)), label
        
        return self.transform(img), label
    

def collate_fn(batch):
    cfp_img = [item[0][0] for item in batch]
    oct_img = [item[0][1] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return (torch.stack(cfp_img), torch.stack(oct_img)), target


## from https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec
def resize_with_pad(image, 
                    new_shape):
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=1.0)
    return image