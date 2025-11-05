from os.path import join

import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import io
# Import custom dataset classes
from .random_dataset import RandDataset
from .episode_dataset import EpiDataset, CategoriesSampler, DCategoriesSampler
from .test_dataset import TestDataset

# Import data transformation utility
from .transforms import data_transform

# Import distributed training utility
from models.utils.comm import get_world_size


class ImgDatasetParam(object):
    """Parameter class for image dataset, storing root paths and embedding types"""
    ## Update directory paths 'imgroot' and "dataroot" according to actual environment
    DATASETS = {
        "imgroot": '/root/shared-nvme/PSVMA/data',  # Root directory for image files
        "dataroot": '/root/shared-nvme/PSVMA/data/xlsa17/data',  # Root directory for .mat annotation files
        "image_embedding": 'res101',  # Type of image embedding (e.g., ResNet-101 features)
        "class_embedding": 'att'  # Type of class embedding (e.g., attribute-based)
    }

    @staticmethod
    def get(dataset):
        """
        Get dataset-specific parameters by appending dataset name to root paths.
        Args:
            dataset (str): Name of target dataset (e.g., 'CUB', 'AwA2', 'SUN')
        Returns:
            dict: Dataset parameters with updated paths
        """
        attrs = ImgDatasetParam.DATASETS
        # Update image root path to include specific dataset subfolder
        attrs["imgroot"] = join(attrs["imgroot"], dataset)
        args = dict(dataset=dataset)
        args.update(attrs)
        return args


def build_dataloader(cfg, is_distributed=False):
    """
    Build training, test-seen, and test-unseen dataloaders based on configuration.
    Args:
        cfg: Configuration object containing dataset/training settings
        is_distributed (bool): Whether to use distributed training (default: False)
    Returns:
        tuple: (train_dataloader, test_unseen_loader, test_seen_loader, data_info_dict)
    """
    # Get dataset-specific parameters (paths, embedding types)
    args = ImgDatasetParam.get(cfg.DATASETS.NAME)
    imgroot = args['imgroot']
    dataroot = args['dataroot']
    image_embedding = args['image_embedding']
    class_embedding = args['class_embedding']
    dataset = args['dataset']

    # Load image path annotations from .mat file
    matcontent = io.loadmat(join(dataroot, dataset, f"{image_embedding}.mat"))
    img_files = np.squeeze(matcontent['image_files'])  # Extract raw image paths

    # --------------------------
    # Process image paths (adapt to dataset-specific folder structure)
    # --------------------------
    new_img_files = []
    for img_file in img_files:
        img_path = img_file[0]  # Convert numpy string to Python string
        # Adjust path based on dataset (CUB/AwA2/SUN have different folder hierarchies)
        if dataset == 'CUB':
            # CUB: Truncate prefix, keep relative path from dataset root
            img_path = join(imgroot, '/'.join(img_path.split('/')[6:]))
        elif dataset == 'AwA2':
            # AwA2: Remove empty string in path segments
            eff_path = img_path.split('/')[5:]
            eff_path.remove('')
            img_path = join(imgroot, '/'.join(eff_path))
        elif dataset == 'SUN':
            # SUN: Truncate longer prefix to match root structure
            img_path = join(imgroot, '/'.join(img_path.split('/')[7:]))
        new_img_files.append(img_path)

    # Convert to numpy array for index-based access
    new_img_files = np.array(new_img_files)  
    # Load and adjust labels (convert to 0-based index)
    label = matcontent['labels'].astype(int).squeeze() - 1  

    # --------------------------
    # Load split information (train/val/test, seen/unseen)
    # --------------------------
    matcontent = io.loadmat(join(dataroot, dataset, f"{class_embedding}_splits.mat"))
    # Convert to 0-based indices for training/validation set
    trainvalloc = matcontent['trainval_loc'].squeeze() - 1  
    # Convert to 0-based indices for test-seen set (classes seen during training)
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1  
    # Convert to 0-based indices for test-unseen set (classes unseen during training)
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1  

    # Extract class attributes and names
    att_name = 'att'
    cls_name = matcontent['allclasses_names']  # Names of all classes in dataset
    attribute = matcontent[att_name].T  # Class attributes (shape: num_classes × num_attributes)

    # --------------------------
    # Prepare training set data
    # --------------------------
    train_img = new_img_files[trainvalloc]  # Image paths for training set
    train_label = label[trainvalloc].astype(int)  # Labels for training set
    train_att = attribute[train_label]  # Attributes corresponding to training labels

    # Reindex training labels to 0-based (in case original labels are non-consecutive)
    train_id, idx = np.unique(train_label, return_inverse=True)
    train_att_unique = attribute[train_id]  # Unique attributes for training classes
    train_clsname = cls_name[train_id]  # Class names for training classes
    num_train = len(train_id)  # Number of unique training classes
    train_label = idx  # Reindexed training labels (0 to num_train-1)
    train_id = np.unique(train_label)  # Confirm reindexed training class IDs

    # --------------------------
    # Prepare test-unseen set data
    # --------------------------
    test_img_unseen = new_img_files[test_unseen_loc]  # Paths for test-unseen images
    test_label_unseen = label[test_unseen_loc].astype(int)  # Original labels for test-unseen

    # Reindex test-unseen labels (start from num_train to avoid overlap with training)
    test_id, idx = np.unique(test_label_unseen, return_inverse=True)
    att_unseen = attribute[test_id]  # Attributes for test-unseen classes
    test_clsname = cls_name[test_id]  # Class names for test-unseen classes
    test_label_unseen = idx + num_train  # Reindexed: num_train to num_train + num_unseen -1
    test_id = np.unique(test_label_unseen)  # Confirm reindexed test-unseen class IDs

    # Combine attributes and IDs for training + test-unseen classes
    train_test_att = np.concatenate((train_att_unique, att_unseen))
    train_test_id = np.concatenate((train_id, test_id))

    # --------------------------
    # Prepare test-seen set data
    # --------------------------
    test_img_seen = new_img_files[test_seen_loc]  # Paths for test-seen images
    test_label_seen = label[test_seen_loc].astype(int)  # Original labels for test-seen

    # Reindex test-seen labels to match training class indices (0 to num_train-1)
    _, idx = np.unique(test_label_seen, return_inverse=True)
    test_label_seen = idx

    # --------------------------
    # Convert key data to PyTorch tensors (for consistency)
    # --------------------------
    att_unseen = torch.from_numpy(att_unseen).float()
    test_label_seen = torch.tensor(test_label_seen)
    test_label_unseen = torch.tensor(test_label_unseen)
    train_label = torch.tensor(train_label)
    att_seen = torch.from_numpy(train_att_unique).float()

    # Package key data info into a dictionary for external use
    res = {
        'label': label,
        'new_img_files': new_img_files,
        'attribute': attribute,
        'train_label': train_label,
        'train_att': train_att,
        'test_label_seen': test_label_seen,
        'test_label_unseen': test_label_unseen,
        'att_unseen': att_unseen,
        'att_seen': att_seen,
        'train_id': train_id,
        'test_id': test_id,
        'train_test_id': train_test_id,
        'train_clsname': train_clsname,
        'test_clsname': test_clsname
    }

    # ================== Debug Information (verify data integrity) ==================
    print(f"[DEBUG] Train set after processing:")
    print(f"  -> train_img: {len(train_img)} samples")
    print(f"  -> train_label: {len(train_label)} samples (min={train_label.min()}, max={train_label.max()})")
    print(f"  -> train_att: {train_att.shape} (samples × attributes)")
    print(f"  -> attribute.shape: {attribute.shape} (total_classes × attributes)")
    print(f"  -> test_unseen_loc: {len(test_unseen_loc)} samples remaining")

    # --------------------------
    # Build Training Dataloader
    # --------------------------
    ways = cfg.DATASETS.WAYS  # Number of classes per episode (for few-shot)
    shots = cfg.DATASETS.SHOTS  # Number of samples per class per episode
    data_aug_train = cfg.SOLVER.DATA_AUG  # Whether to use data augmentation for training
    img_size = cfg.DATASETS.IMAGE_SIZE  # Input image size
    # Get training data transformation pipeline
    transforms = data_transform(data_aug_train, size=img_size)

    # Case 1: Random sampling mode (non-episode based)
    if cfg.DATALOADER.MODE == 'random':
        dataset = RandDataset(train_img, train_att, train_label, transforms)

        if not is_distributed:
            # Non-distributed: Use random sampler + batch sampler (ensure batch size = ways×shots)
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
            batch_size = ways * shots
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, batch_size=batch_size, drop_last=True  # Drop last incomplete batch
            )
            tr_dataloader = DataLoader(
                dataset=dataset,
                num_workers=8,  # Number of worker processes for data loading
                batch_sampler=batch_sampler,
            )
        else:
            # Distributed: Use DistributedSampler (shuffles across processes)
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            batch_size = ways * shots
            tr_dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=8
            )

    # Case 2: Episode sampling mode (for few-shot learning)
    elif cfg.DATALOADER.MODE == 'episode':
        n_batch = cfg.DATALOADER.N_BATCH  # Total number of batches
        ep_per_batch = cfg.DATALOADER.EP_PER_BATCH  # Number of episodes per batch
        dataset = EpiDataset(train_img, train_att, train_label, transforms)

        if not is_distributed:
            # Non-distributed: Use CategoriesSampler (samples episodes)
            sampler = CategoriesSampler(
                train_label,  # All training labels
                n_batch,      # Total batches
                ways,         # Classes per episode
                shots,        # Samples per class
                ep_per_batch  # Episodes per batch
            )
        else:
            # Distributed: Use DCategoriesSampler (distributed episode sampling)
            sampler = DCategoriesSampler(
                train_label,
                n_batch,
                ways,
                shots,
                ep_per_batch
            )

        # Build dataloader with episode sampler
        tr_dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=8,
            pin_memory=True  # Speed up data transfer to GPU
        )

    # --------------------------
    # Build Test Dataloaders (seen/unseen)
    # --------------------------
    data_aug_test = cfg.TEST.DATA_AUG  # Data augmentation for test (usually disabled)
    transforms = data_transform(data_aug_test, size=img_size)  # Test transformation pipeline
    test_batch_size = cfg.TEST.IMS_PER_BATCH  # Batch size for testing

    # Test-Unseen Dataloader (classes unseen during training)
    if not is_distributed:
        tu_data = TestDataset(test_img_unseen, test_label_unseen, transforms)
        tu_loader = DataLoader(
            tu_data,
            batch_size=test_batch_size,
            shuffle=False,  # No shuffle for test
            num_workers=4,
            pin_memory=False
        )
    else:
        tu_data = TestDataset(test_img_unseen, test_label_unseen, transforms)
        # Distributed: Use DistributedSampler (no shuffle for test)
        tu_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tu_data, shuffle=False)
        tu_loader = DataLoader(
            tu_data,
            batch_size=test_batch_size,
            sampler=tu_sampler,
            num_workers=4,
            pin_memory=False
        )

    # Test-Seen Dataloader (classes seen during training)
    if not is_distributed:
        ts_data = TestDataset(test_img_seen, test_label_seen, transforms)
        ts_loader = DataLoader(
            ts_data,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False
        )
    else:
        ts_data = TestDataset(test_img_seen, test_label_seen, transforms)
        ts_sampler = torch.utils.data.distributed.DistributedSampler(dataset=ts_data, shuffle=False)
        ts_loader = DataLoader(
            ts_data,
            batch_size=test_batch_size,
            sampler=ts_sampler,
            num_workers=4,
            pin_memory=False
        )

    # Return training loader, test-unseen loader, test-seen loader, and data info
    return tr_dataloader, tu_loader, ts_loader, res
