import os

import torch
from torchvision import transforms, datasets


# Extended version of ImageFolder to return index and name of image too.
class ImageFolderEx(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        path = self.samples[index][0].split('/')[-1]
        return index, path, sample, target


class TwoCropsTransform:
    """Take two random crops of one image as the query and target."""
    def __init__(self, weak_transform, strong_transform, single_aug=False):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.single_aug = single_aug
        print(self.weak_transform)
        print(self.strong_transform)

    def __call__(self, x):
        q = self.strong_transform(x)
        if self.single_aug:
            return [q, q]
        t = self.weak_transform(x)
        return [q, t]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# Create train loader
def get_train_loader(opt):
    traindir = os.path.join(opt.data, 'train')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    augmentation_strong = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_weak = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    if opt.weak_strong:
        train_dataset = ImageFolderEx(
            traindir,
            TwoCropsTransform(transforms.Compose(augmentation_weak), transforms.Compose(augmentation_strong),
                              opt.single_aug)
        )
    elif opt.weak_weak:
        train_dataset = ImageFolderEx(
            traindir,
            TwoCropsTransform(transforms.Compose(augmentation_weak), transforms.Compose(augmentation_weak),
                              opt.single_aug)
        )
    else:
        train_dataset = ImageFolderEx(
            traindir,
            TwoCropsTransform(transforms.Compose(augmentation_strong), transforms.Compose(augmentation_strong),
                              opt.single_aug)
        )

    if opt.dataset == 'imagenet100':
        subset_classes(train_dataset, num_classes=100)

    print('==> train dataset')
    print(train_dataset)

    # NOTE: remove drop_last
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    return train_loader
