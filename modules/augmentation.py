from torchvision import transforms

class BimodalAugm(object):
    def __init__(
        self,
        img_size=420,
        test = False,
        mean_norm = [0.485, 0.456, 0.406],
        std_norm = [0.229, 0.224, 0.225],
        modality = 'both'
    ):
        self.test = test
        self.modality = modality

        flip_rotate = transforms.v2.Compose([
            transforms.v2.RandomHorizontalFlip(p=0.3),
            transforms.v2.RandomVerticalFlip(p=0.3),
            transforms.v2.RandomApply([transforms.v2.RandomRotation(degrees=(-180,180))], p=0.2),
        ])

        self.norm = transforms.v2.Normalize(mean_norm, std_norm)

        self.cfp_transform = transforms.v2.Compose([
                                    transforms.ToTensor(),
                                    transforms.v2.RandomResizedCrop(
                                        size=(img_size, img_size),  
                                        scale=(0.85, 1.0), antialias=True),
                                    transforms.v2.RandomApply([transforms.v2.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.3),
                                    transforms.v2.RandomGrayscale(p=0.1),
                                    flip_rotate,
                                    transforms.v2.RandomErasing(p=0.1,scale=(0.02, 0.10))
                                ])
        
        self.oct_transform = transforms.v2.Compose([
                                    transforms.ToTensor(),
                                    transforms.v2.Resize((img_size, img_size), antialias=True),
                                    flip_rotate,
                                    transforms.v2.RandomErasing(p=0.1,scale=(0.02, 0.10))
                                ])
        
        self.no_augm = transforms.v2.Compose([
                            transforms.ToTensor(),
                            transforms.v2.Resize((img_size, img_size), antialias=True),
                        ])
    
    def __call__(self, img):

        if self.modality == 'both':
            cfp_img, octs_img = img
            if self.test:
                return self.no_augm(cfp_img), self.no_augm(octs_img)
            return self.cfp_transform(cfp_img), self.norm(self.oct_transform(octs_img))
        
        if self.test:
            return self.no_augm(img)
        
        if self.modality == 'cfp':
            return self.cfp_transform(img)
        
        return self.oct_transform(img)
        # return img