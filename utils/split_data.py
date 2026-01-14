import torch
from torch.utils.data import DataLoader, random_split

def split_data(dataloader, split_length=1024):
    # Assume the dataset used by the input DataLoader is train_dataset
    train_dataset = dataloader.dataset

    # Define the split sizes (e.g., training set and calibration set)
    train_size = len(train_dataset) - split_length
    calib_size = split_length

    # Split the dataset using random_split
    train_subset, calib_subset = random_split(
        train_dataset, [train_size, calib_size]
    )

    # Create DataLoaders for each subset
    train_loader = DataLoader(
        train_subset,
        batch_size=dataloader.batch_size,
        shuffle=True
    )
    calib_loader = DataLoader(
        calib_subset,
        batch_size=dataloader.batch_size,
        shuffle=False
    )

    return train_loader, calib_loader


def split_data_label(dataloader):
    # Collect all data samples and labels from the DataLoader
    all_data, all_label = [], []
    for data, label in dataloader:
        all_data.append(data)
        all_label.append(label)

    # Concatenate all batches into single tensors
    all_data = torch.cat(all_data, dim=0)
    all_label = torch.cat(all_label, dim=0)

    return all_data, all_label
