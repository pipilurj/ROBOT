import copy
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from scipy import stats
from math import inf
import torch.nn.functional as F

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, seed):
    # n -> noise_rate
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed
    print("building dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(len(labels))

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)


    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1


    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)

def uniform_corruption(corruption_ratio, num_classes):
    eye = np.eye(num_classes)
    noise = np.full((num_classes, num_classes), 1/num_classes)
    corruption_matrix = eye * (1 - corruption_ratio) + noise * corruption_ratio
    return corruption_matrix

def flip1_corruption(corruption_ratio, num_classes):
    """mistakes:
        flip in the pair
    """
    P = np.eye(num_classes)
    n = corruption_ratio

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, num_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[num_classes-1, num_classes-1], P[num_classes-1, 0] = 1. - n, n

    return P


def flip2_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_ratio / 2
    return corruption_matrix


def build_dataloader(
        seed=1,
        dataset='cifar10',
        num_meta_total=1000,
        imbalanced_factor=None,
        corruption_type=None,
        corruption_ratio=0.,
        batch_size=100,
        meta_use_train=False,
        meta_batch_size=100,
        return_T=False,
        classes = None,
        train_limits = None,
):

    np.random.seed(seed)
    if dataset == "mnist":
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transforms = train_transforms
    else:
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    dataset_list = {
        'mnist': torchvision.datasets.MNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
    }

    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': noisify_pairflip,
        'flip2': flip2_corruption,
    }

    # train_dataset = dataset_list[dataset](root=f'/home/rpi/prj/data/{dataset}', train=True, download=True, transform=train_transforms)
    # test_dataset = dataset_list[dataset](root=f'/home/rpi/prj/data/{dataset}', train=False, download=True, transform=test_transforms)
    train_dataset = dataset_list[dataset](root=f'../data/{dataset}', train=True, download=True, transform=train_transforms)
    test_dataset = dataset_list[dataset](root=f'../data/{dataset}', train=False, download=True, transform=test_transforms)
    train_dataset_correct = copy.deepcopy(train_dataset)
    if classes is not None:
        train_dataset.classes = classes
        extracted_indices = [np.argwhere(train_dataset.targets == i).squeeze() for i in classes]
        if train_limits is not None:
            extracted_indices = [extracted_indice[:train_limits] for extracted_indice in extracted_indices]
            extracted_indices_cat = np.concatenate(extracted_indices)
        for i in range(len(extracted_indices)):
            train_dataset.targets[extracted_indices[i]] = i
        train_dataset.targets = train_dataset.targets[extracted_indices_cat]
        train_dataset.data = train_dataset.data[extracted_indices_cat]

        test_dataset.classes = classes
        extracted_indices_test = [np.argwhere(test_dataset.targets == i).squeeze() for i in classes]
        extracted_indices_test_cat = np.concatenate(extracted_indices_test)
        for i in range(len(extracted_indices_test)):
            test_dataset.targets[extracted_indices_test[i]] = i
        test_dataset.targets = test_dataset.targets[extracted_indices_test_cat]
        test_dataset.data = test_dataset.data[extracted_indices_test_cat]
    num_classes = len(train_dataset.classes)
    num_meta = int(num_meta_total / num_classes)

    index_to_meta = []
    index_to_train = []

    if imbalanced_factor is not None:
        imbalanced_num_list = []
        sample_num = int((len(train_dataset.targets) - num_meta_total) / num_classes)
        for class_index in range(num_classes):
            imbalanced_num = sample_num / (imbalanced_factor ** (class_index / (num_classes - 1)))
            imbalanced_num_list.append(int(imbalanced_num))
        np.random.shuffle(imbalanced_num_list)
        print(imbalanced_num_list)
    else:
        imbalanced_num_list = None

    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(train_dataset.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])
        index_to_class_for_train = index_to_class[num_meta:]

        if imbalanced_num_list is not None:
            index_to_class_for_train = index_to_class_for_train[:imbalanced_num_list[class_index]]

        index_to_train.extend(index_to_class_for_train)

    true_targets = copy.deepcopy(train_dataset.targets)
    meta_dataset = copy.deepcopy(train_dataset)
    if corruption_type is not None and corruption_type!="instance":
        corruption_matrix = corruption_list[corruption_type](corruption_ratio, num_classes)
        print(corruption_matrix)
        for index in range(len(train_dataset.targets)):
            p = corruption_matrix[train_dataset.targets[index]]
            train_dataset.targets[index] = np.random.choice(num_classes, p=p)
    if corruption_type == "instance":
        new_labels = get_instance_noisy_label(corruption_ratio, train_dataset, train_dataset.targets, num_classes, 3*32*32, 0.1, seed)
        train_dataset.targets = new_labels
        corruption_matrix = None
    train_dataset_ori = copy.deepcopy(train_dataset)
    train_dataset.data = train_dataset.data[index_to_train]
    train_dataset.targets = list(np.array(train_dataset.targets)[index_to_train])
    true_targets_train = np.array(true_targets)[index_to_train]
    # train_dataset.targets = np.array(train_dataset.targets)[index_to_train]
    corrupted_locations = np.argwhere((np.array(true_targets_train) != np.array(train_dataset.targets)) == True).squeeze().tolist()
    correct_locations = np.argwhere((np.array(true_targets_train) == np.array(train_dataset.targets)) == True).squeeze().tolist()
    true_noise_rate = len(corrupted_locations)/(len(corrupted_locations)+len(correct_locations))
    print(f"true_noise_rate {true_noise_rate}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    train_dataloader_unshuffled = DataLoader(train_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)
    train_dataloader_correct = DataLoader(train_dataset_correct, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)
    if meta_use_train:
        meta_dataset = train_dataset_ori
    if num_meta > 0:
        meta_dataset.data = meta_dataset.data[index_to_meta]
        meta_dataset.targets = list(np.array(meta_dataset.targets)[index_to_meta])
        meta_dataloader = DataLoader(meta_dataset, batch_size=meta_batch_size, shuffle=True, pin_memory=True, num_workers=4)
        meta_corrupted_locations = np.argwhere((np.array(true_targets)[index_to_meta] != np.array(meta_dataset.targets)) == True).squeeze().tolist()
        print(f"meta corrput{len(meta_corrupted_locations)}")
        print(f"meta num{len(meta_dataset)}")
        print(f"meta in train {len([x for x in index_to_meta if x in index_to_train])}")
    else:
        meta_dataloader = None
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=False)
    max_class, min_class = np.argmax(imbalanced_num_list), np.argmin(imbalanced_num_list)
    max_locations = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i] == max_class]
    min_locations = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i] == min_class]
    if return_T:
        return train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list,train_dataloader_unshuffled, corrupted_locations, correct_locations, max_locations, min_locations, torch.tensor(true_targets_train), corruption_matrix, train_dataloader_correct
    else:
        return train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list,train_dataloader_unshuffled, corrupted_locations, correct_locations, max_locations, min_locations, torch.tensor(true_targets_train)

