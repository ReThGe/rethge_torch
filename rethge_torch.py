## this file contains custom-code functions for pytorch deeplearning
# containing model training/eval func, results/image plot func, and other help_funcs too
# belongs to: rethge
# created data: 2023/07/02


## imports
# torch related
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# data related

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# system related
import os, gc
import shutil
import pathlib
from pathlib import Path
import random
from typing import Tuple, Dict, List
from timeit import default_timer as timer
from tqdm.auto import tqdm



# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
# utils related funcs
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
def device_picking():
    """
    if GPU is available, using GPU, otherwise use CPU
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} to DeepLearning")
    return device


def check_cuda_cache_and_clean(clean: bool = False):
    """
    run a cuda mem checking, and clean cache when needed
    """


    cached_tensor = torch.cuda.memory_allocated() /1024/1024
    total_cached = torch.cuda.memory_reserved() /1024/1024  

    print(f"current GPU memory occupied by tensors: {cached_tensor} Mb")
    print(f"current GPU memory managed by the caching allocator: {total_cached} Mb")
    print(f"rest GPU mem: {total_cached-cached_tensor} Mb\n")

    if clean:
        gc.collect()
        torch.cuda.empty_cache()
        cached_tensor = torch.cuda.memory_allocated() /1024/1024
        total_cached = torch.cuda.memory_reserved() /1024/1024
        print(f"GPU memory occupied by tensors after clean: {cached_tensor} Mb")
        print(f"GPU memory managed by the caching allocator after clean: {total_cached} Mb")
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
# directory/file manipulate related funcs

def walk_through_dir(dir_path: pathlib.Path):
    """
    know about your dataset dir
    """

    for dirpath, dirname, filenames in os.walk(dir_path):
        print(f"There are {len(dirname)} directories and {len(filenames)} images in '{dirpath}'.")


def rename_get_rid_of_txt_suffix(working_dir: str):

    '''
    working dir should only exist the one type of file, and no folder
    '''
    
    os.chdir(working_dir)
    names=[]
    for i in os.listdir(working_dir):
        n = i.removesuffix('.txt')
        names.append(n)
    
    for i, j in enumerate(os.listdir(working_dir)):
        file_full_dir = f'{working_dir}\{j}'
        rename = f'{working_dir}\{names[i]}'
        os.rename(file_full_dir, rename)


def rename_suffix(working_dir: str,
                           suffix_to_add: str):
    
    """
    add suffix to all the file in a dir
    """
    
    for i in os.listdir(working_dir):
        file_full_dir = f'{working_dir}\{i}'
        rename = f'{file_full_dir}.{suffix_to_add}'
        os.rename(file_full_dir, rename)


def copy_file_to_dir(working_dir: str,
                     aim_dir: str):
    
    """copy all the file to a dir"""
    
    os.chdir(working_dir)
    for file in os.listdir():
        shutil.move(file, aim_dir)


def remove_unused_label(image_dir: str,
                        label_dir: str):
    
    """
    for object detection project data file management
    remove un-used label
    """
    
    label_dir_list = list(Path(label_dir).glob('*.*'))
    name_img = []
    count = 0

    for i in os.listdir(image_dir):

        n = i.removesuffix('.jpg')
        name_img.append(n)

    for names in label_dir_list:
        if names.stem not in name_img:
            os.remove(names)
            count += 1
    print(f"removed {count} unused labels")


def find_missing_label(image_dir: str,
                       label_dir: str) -> list:
    
    """
    for object detection project data file management
    find missed image label
    """
    
    # the stem name of label
    label_stem = []
    image_stem = []
    dir_missing_label = []

    for i in os.listdir(label_dir):
        if i == 'classes.txt':
            continue
        n = i.removesuffix('.txt')
        label_stem.append(n)

    for i in os.listdir(image_dir):
        if i == 'classes.txt':
            continue
        n = i.removesuffix('.jpg')
        image_stem.append(n)

    
    a = [x for x in image_stem if x not in label_stem] 
    for i in a:
        suffix = '.jpg'
        i = f'{i}{suffix}'
        dir = f'{image_dir}\\{i}'
        dir_missing_label.append(Path(dir))
    
    print(f"missing {len(dir_missing_label)} label")

    return dir_missing_label


def adding_nothing_label(image_dir: str,
                         label_dir: str):
    
    """
    for object detection project data file management
    create empty txt file as 'nothing' label
    """
    
    label_name = []
    image_name = []

    for i in os.listdir(label_dir):
        if i == 'classes.txt':
            continue

        nl = i.removesuffix('.txt')
        label_name.append(nl)

    for i in os.listdir(image_dir):
        if i == 'classes.txt':
            continue

        nm = i.removesuffix('.jpg')
        image_name.append(nm)

    compare = [x for x in image_name if x not in label_name] 
    print(f"missing {len(compare)} label\nimage number: {len(image_name)}\nlabel number: {len(label_name)}")
    
    for i in compare:
        suffix = '.txt'
        i = f'{i}{suffix}'
        dir = f'{label_dir}\\{i}'
    
        with open(dir, 'w') as fb:
            fb.close()

    if len(compare) == 0:
        print(f"No label is missing in {label_dir}")
    else: 
        print(f"now having {len(os.listdir(label_dir))} files in folder") 

            
def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
    """
    find the class folder names in a target dir
    
    example:
        classname, class_dict = find_classes(dir) # [anode, cathode, nothing]

    """
    
    classes = sorted(entry.name for entry in os.scandir(dir) if entry.is_dir())
    
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {dir}... please check file structure")
    
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    
    return classes, class_to_idx


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————



# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
# plot related funcs
def plot_trans(img_path_list: List[str],  # img_path_list = list(img_path.glob('*/*/*.jpg'))
               transform: torchvision.transforms, 
               n: int = 3, 
               seed=None):
    """
    select random img from a path list, and using transform, and visualize

    example:
        img_path_list = list(img_path.glob('*/*/*.jpg'))
        transform = transform.Compose([...])
    """

    if seed:
        random.seed(seed)

    random_img_path = random.sample(img_path_list, k=n)
    for p in random_img_path:
        with Image.open(p) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Origin size: {f.size}")
            ax[0].axis(False)

            trans_img = transform(f).permute(1, 2, 0) # we need to change shape for plt
                                        # hwc -> hwc
            ax[1].imshow(trans_img)
            ax[1].set_title(f"transformed img_shape\n: {trans_img.shape}")
            ax[1].axis(False)

            fig.suptitle(f"Class name: {p.parent.stem}", fontsize=16)


def display_random_img(dataset: torch.utils.data.Dataset,
                       classes: List[str] = None,
                       n: int = 10,
                       display_shape: bool = True,
                       seed: int = None):
    '''
    a func to display random img

    Args:
        classes: list of classname,
        n: numbers of img to show
    '''
    
    # nrow=2

    # if not n % 2:
    #     ncol = int(n/2)+1
    # else:
    #     ncol = int(n/2)

    if n > 10:
        n=10
        display_shape = False
        print(f"too many pics to display, max to 10 for display purpose")

    if seed:
        random.seed(seed)

    # get index of random samples
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize=(16,8))

    # loop through idx and plot
    for i, sample_idx in enumerate(random_samples_idx):
        image, label = dataset[sample_idx][0].permute(1,2,0), dataset[sample_idx][1]

        plt.subplot(1, n, i+1)
        plt.imshow(image)
        plt.axis(False)

        if classes:
            title = f"Class: {classes[label]}"
            if display_shape:
                title += f"\nshape: {image.shape}"
        plt.title(title)


def plot_lr(results: Dict[str, List[float]]):
    """
    this funcs plot a lr_scheduler's curve varying with epochs when a training is over
    """
    lr = results['learning rate']
    epochs = range(len(results['learning rate']))

    plt.figure(figsize=(7,7))
    plt.plot(epochs, lr, label='learning rate')
    plt.title('learning rate scheduler')
    plt.xlabel('Epochs')
    plt.legend()

def plot_loss_curves(results: Dict[str, List[float]] or Path):
    """
    results is a dict and will be like: 
    {'train_loss': [...],
        'train_acc': [...],
        'test_loss': [...],
        'test_acc': [...]}
    """
    if type(results) != dict:
        results = pd.read_csv(results)
        results = results.iloc[:, 1:] # row, col
        results = results.to_dict("list")
    
    else:
        pass

    loss = results['train_loss']
    test_loss = results['test_loss']

    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15,7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_acc')
    plt.plot(epochs, test_accuracy, label='test_acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


def pred_single_img(Model: torch.nn.Module,
                    image_path: str,
                    class_name: List[str] = None,
                    transforms = None,
                    device: torch.device = torch.device('cpu')
                    ):
    """
    show a img's pred results
    """

    image_done = torchvision.io.read_image(image_path).type(torch.float).to(device) / 255. 
    Model.to(device)

    if transforms:
        image_done = transforms(image_done).unsqueeze(0).to(device)

    Model.eval()
    with torch.inference_mode():
        pred = Model(image_done)
        pred_probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1)

    plt.imshow(image_done.squeeze().permute(1,2,0))
    title = f'Pred: {class_name[pred_class.cpu()]} | Probs: {pred_probs.max().cpu():.4f}'
    plt.title(title)
    plt.axis(False)

    return pred_probs


def plot_conf_mat(predictions: List[int],
                  num_classes: int,
                  classname,
                  dataset_imagefolder: datasets.ImageFolder,
                  task: str = 'multiclass'):

    confmat = ConfusionMatrix(num_classes=num_classes,
                              task=task)
    
    confmat_tensor = confmat(preds=predictions,
                             target=torch.tensor(dataset_imagefolder.targets))
    
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # plt like working with np
        class_names=classname,
        figsize=(10,7))


def plot_patch_img(img: torch.Tensor,
                   img_size: int = 224,
                   patch_size: int = 16,):
        
        """this is for ViT demonstrate"""
        

        # Setup hyperparameters and make sure img_size and patch_size are compatible

        num_patches = img_size/patch_size 
        assert img_size % patch_size == 0, "Image size must be divisible by patch size" 
        
        print(f"Number of patches per row: {num_patches}\
                \nNumber of patches per column: {num_patches}\
                \nTotal patches: {num_patches*num_patches}\
                \nPatch size: {patch_size} pixels x {patch_size} pixels")

        image_permuted = img.permute(1, 2, 0)
        # Create a series of subplots
        fig, axs = plt.subplots(nrows=img_size // patch_size, # need int not float
                                ncols=img_size // patch_size, 
                                figsize=(num_patches, num_patches),
                                sharex=True,
                                sharey=True)

        # Loop through height and width of image
        for i, patch_height in enumerate(range(0, img_size, patch_size)): # iterate through height
                for j, patch_width in enumerate(range(0, img_size, patch_size)): # iterate through width
                        
                        # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
                        axs[i, j].imshow(image_permuted[patch_height:patch_height+patch_size, # iterate through height 
                                             patch_width:patch_width+patch_size, # iterate through width
                                             :]) # get all color channels
                        
                        # Set up label information, remove the ticks for clarity and set labels to outside
                        axs[i, j].set_ylabel(i+1, 
                                        rotation="horizontal", 
                                        horizontalalignment="right", 
                                        verticalalignment="center"
                                        ) 
                        axs[i, j].set_xlabel(j+1) 
                        axs[i, j].set_xticks([])
                        axs[i, j].set_yticks([])
                        axs[i, j].label_outer()

        plt.show()


def plot_5_feature_map(img_conv_out: torch.Tensor,
                       embedding_size: int = 768,):
    """
    Plot random 5 convolutional feature maps, for ViT
    """
    random_indexes = random.sample(range(0, embedding_size), k=5) # pick 5 numbers between 0 and the embedding size
    print(f"Showing random convolutional feature maps from indexes: {random_indexes}")

    # Create plot
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 12))

    # Plot random image feature maps
    for i, idx in enumerate(random_indexes):
        img_feature_map = img_conv_out[:, idx, :, :] # index on the output tensor of the convolutional layer
        axs[i].imshow(img_feature_map.squeeze().detach().numpy())
        axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[]);

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————




# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
## Data_load related

# custom ImageFolder
class RTG_RAM_DataSet(Dataset):
    def __init__(self,
                 dir: str,
                 transform=None):
        super().__init__()

        """
        this is a custom ImageFolder of pytorch
        load your data into RAM in advance
        can boost the training process
        """

        self.paths = list(Path(dir).glob("*/*.jpg")) # pathlib.Path

        self.transform = transform

        self.classes, self.class_idx = find_classes(dir)

    def load_image(self, index: int) -> Image.Image:
        """Open an image via a path and return it"""

        image_path = self.paths[index]
        return Image.open(image_path)
    
    # overwrite __len__()
    def __len__(self) -> int:
        """return the total num of samples."""
        return len(self.paths)
    
    # overwrite __getitem__()
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """return one sample of data, and label like (X, y)."""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_idx[class_name]

        # transformation if necessary
        if self.transform:
            return self.transform(img), class_idx # return data+label (X,y)
        else: 
            return img, class_idx
        

def create_dataloaders(
    train_dir: str, 
    valid_dir: str, 
    transform: transforms.Compose,
    batch_size: int, 
    test_transform: transforms.Compose = None, 
    num_workers: int=0,
    test_dir: str = None,
    pin_mem: bool = True
):
  
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.

  """
  # Use ImageFolder to create dataset(s)
  train_data = RTG_RAM_DataSet(train_dir, transform=transform)
  valid_data = RTG_RAM_DataSet(valid_dir, transform=transform)

  if test_dir :
    test_data = RTG_RAM_DataSet(test_dir, transform=test_transform)

    test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_mem,)
  else:
    pass

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=pin_mem,
  )

  valid_dataloader = DataLoader(
      valid_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=pin_mem,
  )


  if test_dir:
    return train_dataloader, valid_dataloader, test_dataloader, class_names
  else:
    return train_dataloader, valid_dataloader, class_names


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————



# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
## model related

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """Prints and return time cost."""
    total_time = end - start
    print(f"train time on {device}: {total_time:.3f} seconds")
    return total_time


def general_train_setup(Model: nn.Module,
                        train_path: Path, 
                        valid_path: Path, 
                        test_path: Path,
                        transform: transforms, 
                        test_transform: transforms,
                        batch_size: int = 8, 
                        num_worker: int = 8,  # cpu cores
                        init_lr: float = 0.01,
                        linearLR_factor: float = 0.1,
                        expLR_gamma: float = 0.95,
                        constLR_factor: float = 0.2,
                        mileston1: int = 30,
                        mileston2: int = 60,
                        epochs: int = 100
                        ):
    
    """
    quick setup for a training
    
    Returns:
        a dict that contain dataloader, lr_scheduler(if needed), loss_fn, optimizing_func, classnames
    """

    train_dataloader, valid_dataloader, test_dataloader, class_name = create_dataloaders(train_dir=train_path,
                                                                                        valid_dir=valid_path,
                                                                                        test_dir=test_path,
                                                                                        test_transform=test_transform,
                                                                                        batch_size=batch_size,
                                                                                        num_workers=num_worker,
                                                                                        transform=transform,
                                                                                        pin_mem=True)
    
    last = epochs-mileston2

    loss_fn = torch.nn.CrossEntropyLoss()
    optima = torch.optim.AdamW(params=Model.parameters(), lr=init_lr, eps=1e-3) # 0.01

    scheduler1 = torch.optim.lr_scheduler.LinearLR(optima, start_factor=linearLR_factor)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optima, gamma=expLR_gamma)  # also need to tune gamma here
    scheduler3 = torch.optim.lr_scheduler.ConstantLR(optima, factor=constLR_factor, total_iters=last)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optima, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[mileston1, mileston2])

    if test_path:
        return {'train_dataloader': train_dataloader, 
                'valid_dataloader': valid_dataloader, 
                'test_dataloader': test_dataloader,
                'class_name': class_name, 
                'scheduler': scheduler, 
                'loss_fn': loss_fn, 
                'optima': optima}
    else:
        return {'train_dataloader': train_dataloader, 
                'valid_dataloader': valid_dataloader, 
                'class_name': class_name, 
                'scheduler': scheduler, 
                'loss_fn': loss_fn, 
                'optima': optima}


def train_step(Model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optima: torch.optim.Optimizer,
               #accuracy_fn,
               device: torch.device = torch.device("cpu")):
    """
    Performs a training with model trying to learn on data loader.
    train a single step
    """

    train_loss, train_acc = 0, 0

    Model.to(device)
    # with torch.cuda.device(device=device): # this is useless
    Model.train()

    for _, (X, y) in enumerate(data_loader):
    # batch       
        X, y = X.to(device), y.to(device)

        y_pred_t = Model(X) 
        loss_t = loss_fn(y_pred_t, y)
        loss_t.backward()
        optima.step() # updata params per batch, not per epoch
        
        optima.zero_grad(set_to_none=True)
            # for param in Model.parameters():
            #     param.grad = None
        
        train_loss += loss_t.item() # .item() turn single tensor into a single scaler
        y_pred_t_class = torch.argmax(y_pred_t, dim=1)
        train_acc += torch.eq(y_pred_t_class, y).sum().item()/len(y_pred_t) * 100
 

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    # print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}%")
    return train_acc, train_loss


def test_step(Model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              #accuracy_fn,
              device: torch.device = torch.device("cpu")):
    '''test/valid a single step'''

    test_loss, test_acc = 0, 0

    Model.to(device)

    Model.eval()
    with torch.inference_mode():
        for X, y in data_loader:

            X, y = X.to(device), y.to(device)

            y_pred_e = Model(X)
            test_loss += loss_fn(y_pred_e, y).item()

            y_pred_e_labels = y_pred_e.argmax(dim=1)
            test_acc += torch.eq(y_pred_e_labels, y).sum().item()/len(y_pred_e) * 100

            # test_acc += accuracy_fn(y_true=y,
            #                         y_pred=y_pred_e.argmax(dim=1))
            
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        # print(f"Test loss: {test_loss:.4F} | Test acc: {test_acc:.4F}%\n")
        return test_acc, test_loss   


def train_test_loop(Model: torch.nn.Module,
                    train_loader: torch.utils.data.DataLoader,
                    test_loader: torch.utils.data.DataLoader,
                    epochs: int,
                    optima: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler = None,
                    #accuracy_fn,
                    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
                    device: torch.device = torch.device("cpu")):
        
        if scheduler is not None:
            results = {'train_loss': [],
                    'train_acc': [],
                    'test_loss': [],
                    'test_acc': [],
                    'learning rate': []}
        else:
            results = {'train_loss': [],
                    'train_acc': [],
                    'test_loss': [],
                    'test_acc': [],}

        Model.to(device)
        time_start = timer()

        for ep in tqdm(range(epochs)):

            train_acc, train_loss = train_step(Model=Model,
                        data_loader=train_loader,
                        loss_fn=loss_fn,
                        optima=optima,
                        device=device)
    
            test_acc, test_loss = test_step(Model=Model,
                        data_loader=test_loader,
                        loss_fn=loss_fn,
                        device=device)
            
            if scheduler is not None:
                current_lr = optima.param_groups[0]['lr']
                results['learning rate'].append(current_lr)
                scheduler.step()
            
            print(f"Epoch: {ep+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}"
                    )
            
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['test_loss'].append(test_loss)
            results['test_acc'].append(test_acc)

        time_end = timer()
        _ = print_train_time(start=time_start,
                         end=time_end,
                         device=device)
        
        return results


def train_test_loop_with_amp(Model: torch.nn.Module,
                            train_loader: torch.utils.data.DataLoader,
                            test_loader: torch.utils.data.DataLoader,
                            epochs: int,
                            optima: torch.optim.Optimizer,
                            scheduler: torch.optim.lr_scheduler = None,
                            loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
                            device: torch.device = torch.device("cpu")):
    
    """ 
    using AMP to training
    """
        
    if scheduler is not None:
        results = {'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': [],
                'learning rate': []}
    else:
        results = {'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': [],}


    # train_loss, train_acc = 0, 0

    Model.to(device)
    Model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    time_start = timer()
    for ep in tqdm(range(epochs)):

        train_loss, train_acc = 0, 0 #?? maybe to avoid nan?

        for X, y in train_loader:     
            X, y = X.to(device), y.to(device)

            optima.zero_grad(set_to_none=True)
            # for param in Model.parameters():
            #     param.grad = None

            with torch.autocast(device_type=str(device), dtype=torch.float16):

                y_pred_t = Model(X) 
                loss_t = loss_fn(y_pred_t, y)
                
            # or maybe we should move this two line inside of AMP block? 
            train_loss += loss_t.item() # .item() turn single tensor into a single scaler
            y_pred_t_class = torch.argmax(y_pred_t, dim=1)
            train_acc += torch.eq(y_pred_t_class, y).sum().item()/len(y_pred_t) * 100

            scaler.scale(loss_t).backward() # none type
            
            scaler.unscale_(optima)
     
            torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=0.1)
        
            scaler.step(optima)  
            scaler.update()

            # loss_t.backward()
            # optima.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        if train_acc > 100:
            train_acc = 100.0000

        test_acc, test_loss = test_step(Model=Model,
                    data_loader=test_loader,
                    loss_fn=loss_fn,
                    device=device)
        
        if scheduler is not None:
            optima.zero_grad(set_to_none=True)
            optima.step()
            current_lr = optima.param_groups[0]['lr']
            results['learning rate'].append(current_lr)
            scheduler.step()
        
        print(f"Epoch: {ep+1} | "
                f"train_loss: {train_loss:.4f} | "  # nan???
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "    # nan???
                f"test_acc: {test_acc:.4f}"
                )
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        # gc.collect()
        # torch.cuda.empty_cache()

    time_end = timer()
    print_train_time(start=time_start,
                        end=time_end,
                        device=device)
    
    return results



def eval_model(Model: torch.nn.Module,
               eval_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
               show: bool = True,
               device: torch.device = torch.device("cpu")):
    '''
    eval model prediction results, return loss, acc, pred_tensor
    pred_tensor is for the plot of confusion matrix
    '''
    loss = 0
    acc = 0
    preds = []

    Model.to(device)
    Model.eval()
    with torch.inference_mode():
        for X, y in tqdm(eval_loader):
            X, y = X.to(device), y.to(device)

            raw_logits = Model(X)

            loss += loss_fn(raw_logits, y).item()
            pred_label = torch.argmax(raw_logits, dim=1)
            
            prediction = torch.argmax(raw_logits.squeeze(0), dim=1) # using this for confusion matrix
            preds.append(prediction.cpu())
            
            acc += torch.eq(pred_label, y).sum().item()/len(raw_logits) * 100

        loss /= len(eval_loader)
        acc /= len(eval_loader)
        
    predictions_tensor = torch.cat(preds)
    
    if show:
        print(f"Model: {Model.__class__.__name__}")
        print(f"Eval loss: {loss:.4F} | Eval acc: {acc:.4F}%\n")
    return loss, acc, predictions_tensor


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
## result saving
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the whole model, not only the state_dict(), so that we don't have to init model structure instance everytime
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model, # .state_dict(), 
             f=model_save_path)


def save_results(results: Dict[str, List[float]],
                 path_and_filename: str):
    '''save Dict results into csv format'''

    print(f"[INFO] Saving results to: {path_and_filename}")   
    df = pd.DataFrame(results)
    df.to_csv(path_and_filename, index=False)


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
## result analyze related 

def pred_wrong_and_store(path: Path, # class1..classn/img.jpg
                   Model,
                   transform,
                   class_names,
                   top_num: int = 5,
                   show: bool = True,
                   device: torch.device = torch.device('cpu')):
    """
    preds some img on a model and store the results
    and also grab and plot some most wrong examples

    Returns:
        a sorted pandas dataframe
    """

    pred_list = []

    # first, get a list contain every single img path
    img_path_list = list(Path(path).glob("*/*.jpg")) 


    for path in tqdm(img_path_list):

        # a empty dict to store every img result
        pred_dict = {}

        # get sample path
        pred_dict['img_path'] = path

        # get class name
        class_name = path.parent.stem
        pred_dict["class_names"] = class_name

        start_time = timer()

        # get predictions
        img = Image.open(path)
        transformed_img = transform(img).unsqueeze(0).to(device)

        Model.to(device)
        Model.eval()
        with torch.inference_mode():
            pred_logits = Model(transformed_img)
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)
            pred_class = class_names[pred_label.cpu()]

            pred_dict["pred_probs"] = pred_probs.unsqueeze(0).max().cpu().item() # make sure result back to cpu
            pred_dict["pred_class"] = pred_class # convient for plot

            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time-start_time, 4)

        pred_dict['correct'] = class_name == pred_class

        pred_list.append(pred_dict)
    
    pred_df = pd.DataFrame(pred_list)
    sorted_pred_df = pred_df.sort_values(by=['correct', 'pred_probs'], ascending=[True, False])

    if show:
        most_wrong = sorted_pred_df.head(n=top_num)

        for row in most_wrong.iterrows():
            data_row = row[1]
            img_path = data_row[0] 
            true_label = data_row[1]
            pred_prob = data_row[2]
            pred_class = data_row[3]

            # plot img
            img = torchvision.io.read_image(str(img_path)) # read to tensor
            plt.figure()
            plt.imshow(img.permute(1, 2, 0)) # h x w x c
            plt.title(f"True: {true_label} | Pred: {pred_class} | Prob: {pred_prob}")
            plt.axis(False);
    else:
        pass
    
    return sorted_pred_df


def check_model_size(path, show=True):
    """check a model's size"""

    size = Path(path).stat().st_size // (1024*1024)
    if show:
        print(f"model size: {size:.3f} MB")

    return size


def general_test(Model, 
                 model_path, 
                 class_name, 
                 manual_transforms, 
                 test_path, loss_fn, 
                 valid_loader):
    
    """
    run a general test on a model
    including model_size, params, loss and acc on test set, pred_time and so on

    Returns:
        a dict
    """

    stat = {}
    print(f'[INFO] running general test on: {Model._get_name()}')

    model_size = check_model_size(model_path, show=False)
    print('size check ... done')
    model_params = sum(torch.numel(param) for param in Model.parameters())
    print('params check ... done')
    loss, acc, _ = eval_model(Model, valid_loader, loss_fn, show=False)   
    print('valid evaluate ... done')
    pred_df = pred_wrong_and_store(test_path, Model, manual_transforms, class_name, show=False)
    print('prediction test ... done')
    average_time_per_pred = round(pred_df.time_for_pred.mean(), 4)
    print('predict time calculate ... done')
    test_acc = pred_df.correct.value_counts()[0]*100/len(pred_df)
    print('real accurate calculate ... done')

    stat['valid_loss'] = loss
    stat['valid_acc'] = acc
    stat['test_acc'] = test_acc
    stat['number_of_parameters'] = model_params
    stat['model_size (MB)'] = model_size
    stat['time_per_pred_cpu'] = average_time_per_pred

    print("test results:")
    print(stat)

    return stat
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————