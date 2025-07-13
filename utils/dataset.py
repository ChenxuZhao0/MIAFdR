import os
import torch
import numpy as np
import torchvision
import glob
import pickle

from torch.utils.data import Subset, DataLoader, random_split, ConcatDataset, TensorDataset
from torchvision import transforms, datasets
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def construct_datasets(args, dataset, data_path, load=False):
    if dataset == 'CIFAR10':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std), ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std), ])

        train_data = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        test_data = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

        if load == False:
            train_labels = np.array(train_data.targets)
            train_target_indices, train_source_indices = train_test_split(range(len(train_labels)), train_size=args.train_size, stratify=train_labels)
            test_labels = np.array(test_data.targets)
            test_target_indices, test_source_indices = train_test_split(range(len(test_labels)), train_size=args.train_size, stratify=test_labels)

            data_indices = {'train_target_indices': train_target_indices, 'train_source_indices': train_source_indices, 'test_target_indices': test_target_indices, 'test_source_indices': test_source_indices}
            file_path = os.path.join(data_path, args.dataset + '_' + str(args.train_size) + '_' + args.net + '_data.pkl')
            with open(file_path, 'wb') as file:
                pickle.dump(data_indices, file)
        else:
            file_path = os.path.join(data_path, args.dataset + '_' + str(args.train_size) + '_' + args.net + '_data.pkl')
            with open(file_path, 'rb') as file:
                data_indices = pickle.load(file)
            train_target_indices, train_source_indices, test_target_indices, test_source_indices = data_indices['train_target_indices'], data_indices['train_source_indices'], data_indices['test_target_indices'], data_indices['test_source_indices']

        train_target_set = torch.utils.data.Subset(train_data, train_target_indices)
        train_source_set = torch.utils.data.Subset(train_data, train_source_indices)
        test_target_set = torch.utils.data.Subset(test_data, test_target_indices)
        test_source_set = torch.utils.data.Subset(test_data, test_source_indices)

    elif args.dataset == 'CIFAR100':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std), ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std), ])

        train_data = CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        test_data = CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

        if load == False:
            train_labels = np.array(train_data.targets)
            train_target_indices, train_source_indices = train_test_split(range(len(train_labels)), train_size=args.train_size, stratify=train_labels)
            test_labels = np.array(test_data.targets)
            test_target_indices, test_source_indices = train_test_split(range(len(test_labels)), train_size=args.train_size, stratify=test_labels)

            data_indices = {'train_target_indices': train_target_indices, 'train_source_indices': train_source_indices, 'test_target_indices': test_target_indices, 'test_source_indices': test_source_indices}
            file_path = os.path.join(data_path, args.dataset + '_' + str(args.train_size) + '_' + args.net + '_data.pkl')
            with open(file_path, 'wb') as file:
                pickle.dump(data_indices, file)
        else:
            file_path = os.path.join(data_path, args.dataset + '_' + str(args.train_size) + '_' + args.net + '_data.pkl')
            with open(file_path, 'rb') as file:
                data_indices = pickle.load(file)
            train_target_indices, train_source_indices, test_target_indices, test_source_indices = data_indices['train_target_indices'], data_indices['train_source_indices'], data_indices['test_target_indices'], data_indices['test_source_indices']

        train_target_set = torch.utils.data.Subset(train_data, train_target_indices)
        train_source_set = torch.utils.data.Subset(train_data, train_source_indices)
        test_target_set = torch.utils.data.Subset(test_data, test_target_indices)
        test_source_set = torch.utils.data.Subset(test_data, test_source_indices)

    elif args.dataset == 'TinyImageNet':
        data_mean = (0.4802, 0.4481, 0.3975)
        data_std = (0.2302, 0.2265, 0.2262)
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std), ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std), ])

        train_data = TinyImageNet(root=data_path, split='train', transform=transform_train, classes="firsthalf")
        test_data = TinyImageNet(root=data_path, split='val', transform=transform_test, classes="firsthalf")

        if load == False:
            train_labels = np.array(train_data.targets)
            train_target_indices, train_source_indices = train_test_split(range(len(train_labels)), train_size=args.train_size, stratify=train_labels)
            test_labels = np.array(test_data.targets)
            test_target_indices, test_source_indices = train_test_split(range(len(test_labels)), train_size=args.train_size, stratify=test_labels)

            data_indices = {'train_target_indices': train_target_indices, 'train_source_indices': train_source_indices, 'test_target_indices': test_target_indices, 'test_source_indices': test_source_indices}
            file_path = os.path.join(data_path, args.dataset + '_' + str(args.train_size) + '_' + args.net + '_data.pkl')
            with open(file_path, 'wb') as file:
                pickle.dump(data_indices, file)
        else:
            file_path = os.path.join(data_path, args.dataset + '_' + str(args.train_size) + '_' + args.net + '_data.pkl')
            with open(file_path, 'rb') as file:
                data_indices = pickle.load(file)
            train_target_indices, train_source_indices, test_target_indices, test_source_indices = data_indices['train_target_indices'], data_indices['train_source_indices'], data_indices['test_target_indices'], data_indices['test_source_indices']

        train_target_set = torch.utils.data.Subset(train_data, train_target_indices)
        train_source_set = torch.utils.data.Subset(train_data, train_source_indices)
        test_target_set = torch.utils.data.Subset(test_data, test_target_indices)
        test_source_set = torch.utils.data.Subset(test_data, test_source_indices)

    return train_target_set, train_source_set, test_target_set, test_source_set


class CIFAR10(datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""
    def __getitem__(self, index):
        """
        Returns: (image, target, idx) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_target(self, index):
        """
        Returns: (target, idx) where target is class_index of the target class.
        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class CIFAR100(datasets.CIFAR100):
    """Super-class CIFAR100 to return image ids with images."""
    def __getitem__(self, index):
        """
        Returns: (image, target, idx) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_target(self, index):
        """
        Returns: (target, idx) where target is class_index of the target class.
        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class TinyImageNet(torch.utils.data.Dataset):
    """
    Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    """

    EXTENSION = 'JPEG'
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = 'wnids.txt'
    VAL_ANNOTATION_FILE = 'val_annotations.txt'
    CLASSES = 'words.txt'

    def __init__(self, root, split='train', transform=None, target_transform=None, classes="all"):
        """Init with split, transform, target_transform. use --cached_dataset data is to be kept in memory."""
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % self.EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping

        # build class label - number mapping
        count = 0
        with open(os.path.join(self.root, self.CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, self.EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # Build class names
        label_text_to_word = dict()
        with open(os.path.join(root, self.CLASSES), 'r') as file:
            for line in file:
                label_text, word = line.split('\t')
                label_text_to_word[label_text] = word.split(',')[0].rstrip('\n')
        self.classes = [label_text_to_word[label] for label in self.label_texts]

        # Prepare index - label mapping
        self.targets = [self.labels[os.path.basename(file_path)] for file_path in self.image_paths]

        if classes == "firsthalf":
            idx = np.where(np.array(self.targets) < 100)[0]
        elif classes == "lasthalf":
            idx = np.where(np.array(self.targets) >= 100)[0]
        else:
            idx = np.arange(len(self.targets))
        self.image_paths = [self.image_paths[i] for i in idx]
        self.targets = [self.targets[i] for i in idx]
        self.targets = [t - min(self.targets) for t in self.targets]

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return a triplet of image, label, index."""
        file_path, target = self.image_paths[index], self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        img = Image.open(file_path)
        img = img.convert("RGB")
        img = self.transform(img) if self.transform else img
        if self.split == 'test':
            return img, None
        else:
            return img, target


    def get_target(self, index):
        """Return only the target and its id."""
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index



class CustomTransformDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = Subset(dataset, indices)
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


def data_random_split(args, logging, train_data, test_data, batch_size=32):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),])

    indices = torch.randperm(50000)
    train_indices = indices[:25000]
    test_indices_half = indices[25000:]

    augmented_data = CustomTransformDataset(train_data, train_indices, transform=transform_train)

    test_data_half = CustomTransformDataset(train_data, test_indices_half, transform=transform_test)

    no_augmented_data = ConcatDataset([test_data_half, test_data])

    # A - Private Train
    pri_tra, temp = random_split(augmented_data, [5000, 20000])
    pri_tra_loader = torch.utils.data.DataLoader(pri_tra, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    #  - Shadow Train
    shad_1_tra, temp = random_split(temp, [5000, 15000])
    shad_1_tra_loader = torch.utils.data.DataLoader(shad_1_tra, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    shad_2_tra, temp = random_split(temp, [5000, 10000])
    shad_2_tra_loader = torch.utils.data.DataLoader(shad_2_tra, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    shad_3_tra, shad_4_tra = random_split(temp, [5000, 5000])
    shad_3_tra_loader = torch.utils.data.DataLoader(shad_3_tra, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    shad_4_tra_loader = torch.utils.data.DataLoader(shad_4_tra, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    # E - Calibration Dataset
    non_memebr_cali_dataset, temp = random_split(no_augmented_data, [10000, 25000])

    # B - Private Test
    pri_test, temp = random_split(temp, [5000, 20000])
    pri_test_loader = torch.utils.data.DataLoader(pri_test, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    # - Shadow Test
    shad_1_tes, temp = random_split(temp, [5000, 15000])
    shad_1_tes_loader = torch.utils.data.DataLoader(shad_1_tes, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    shad_2_tes, temp = random_split(temp, [5000, 10000])
    shad_2_tes_loader = torch.utils.data.DataLoader(shad_2_tes, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    shad_3_tes, shad_4_tes = random_split(temp, [5000, 5000])
    shad_3_tes_loader = torch.utils.data.DataLoader(shad_3_tes, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    shad_4_tes_loader = torch.utils.data.DataLoader(shad_4_tes, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    # C - Sample Test data
    if args.alpha == 0.25:
        x, y = 3600, 1200
    elif args.alpha == 0.5:
        x, y = 3600, 3600
    elif args.alpha == 0.75:
        x, y = 1200, 3600

    test_member, _ = random_split(pri_tra, [x, 5000-x])
    test_member_loader = torch.utils.data.DataLoader(test_member, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    test_non_member, _ = random_split(pri_test, [y, 5000-y])
    test_non_member_loader = torch.utils.data.DataLoader(test_non_member, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    shadow_tra_loader_list = [shad_1_tra_loader,shad_2_tra_loader,shad_3_tra_loader,shad_4_tra_loader]
    shadow_test_loader_list = [shad_1_tes_loader,shad_2_tes_loader,shad_3_tes_loader,shad_4_tes_loader]

    return pri_tra_loader,pri_test_loader,non_memebr_cali_dataset,shadow_tra_loader_list,shadow_test_loader_list,test_member_loader,test_non_member_loader