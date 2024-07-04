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
import math
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# system related
import os, gc
import sys
import shutil
import pathlib
from pathlib import Path
import random
import string
from collections import Counter
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
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: 128"

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


def iou(box_pred:torch.tensor, box_label:torch.tensor, box_format:str='midpoint'):
    # box_pred shape is (N, 4) where N is the num of bboxes (2, 4): [[x1, y1, x2, y2], [...]]
    # box_label shape is (N, 4)
    if box_format == 'midpoint': # xywh
        b1_x1 = box_pred[..., 0:1] - box_pred[..., 2:3] / 2 # x - w/2
        b1_y1 = box_pred[..., 1:2] - box_pred[..., 3:4] / 2
        b1_x2 = box_pred[..., 0:1] + box_pred[..., 2:3] / 2 # x + w/2
        b1_y2 = box_pred[..., 1:2] + box_pred[..., 3:4] / 2
        b2_x1 = box_label[..., 0:1] - box_label[..., 2:3] / 2
        b2_y1 = box_label[..., 1:2] - box_label[..., 3:4] / 2
        b2_x2 = box_label[..., 0:1] + box_label[..., 2:3] / 2
        b2_y2 = box_label[..., 1:2] + box_label[..., 3:4] / 2

    if box_format == 'corners':
        b1_x1 = box_pred[..., 0:1] # (N, 1), keep the shape
        b1_y1 = box_pred[..., 1:2]
        b1_x2 = box_pred[..., 2:3]
        b1_y2 = box_pred[..., 3:4]

        b2_x1 = box_label[..., 0:1]
        b2_y1 = box_label[..., 1:2]
        b2_x2 = box_label[..., 2:3]
        b2_y2 = box_label[..., 3:4]

    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    inter_aera = (x2-x1).clamp(0) * (y2-y1).clamp(0) # if for the case when they do not intersect
    b1_aera = abs((b1_x2-b1_x1)*(b1_y2-b1_y1))
    b2_aera = abs((b2_x2-b2_x1)*(b2_y2-b2_y1))

    return inter_aera / (b1_aera + b2_aera - inter_aera + 1e-6)


def nms(pred, iou_threshold, prob_threshold, box_format='corners'): # pred inputs is bbox
    # pred = [[class, prob, x1,y1,x2,y2], [], []] # 3 bbox
    assert type(pred)==list
    bboxes = [box for box in pred if box[1] > prob_threshold] # take high prob boxes
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # lambda variable... : operation, sort the prob_thres
    bboxes_after_nms = []

    while bboxes: # we compare every boxes, every time pop one out
        chosen_box = bboxes.pop(0)
        bboxes = [box 
                  for box in bboxes 
                  if box[0] != chosen_box[0] or # if not the same class 
                  iou(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format) < iou_threshold] # if other box has higer iou than (let's say) 0.5
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mAP(pred_boxes, true_boxes, num_classes,
        iou_threshold=0.5, box_format='corners'):
    
    # pred_boxes = list: [[img_idx, class, prob, x1,y1,x2,y2], [...], [...]]
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes): # we check each class detection results(0-19)
        detections = []
        ground_truths = []

        for detection in pred_boxes: # how many box detected as class 0? [[img_0, class_0, prob, x1,y1,x2,y2], [...], ...]
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes: # the true box on class 0 [[img_idx, class_0, prob, x1,y1,x2,y2], [...], ...], (do we have prob on truth?)
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        # how many true boxes for all img on class 0? Counter([00001122333445522666...])
        # img0 has 3 box of class 0
        # img1 has 5 box of class 0
        # amount_bboxes = {0: 3, 1: 5} for class 0
        amount_bboxes = Counter([gt[0] for gt in ground_truths]) 

        for k, v in amount_bboxes.items(): # for the convient of tracking and checking of true boxes
            amount_bboxes[k] = torch.zeros(v) # amount_bboxes = {0: torch.tensor([0,0,0]), 1: torch.tensor([0,0,0,0,0])}
                                              # and the pred might differ with {0: 4, 1: 4...}
        
        detections.sort(key=lambda x: x[2], reverse=True) # by prob
        TP = torch.zeros((len(detections))) # (00000000000000000000000)
        FP = torch.zeros((len(detections))) # (00000000000000000000000)
        total_ture_boxes = len(ground_truths) # (0000000000000000)

        for detection_idx, detection in enumerate(detections): # 0, [img_0, class_0, prob, x1,y1,x2,y2], start at img 0, pred box 1:
            ground_truth_box = [bbox 
                                for bbox in ground_truths 
                                if bbox[0] == detection[0]] # we take all box on one img per time to compare
            # in this case, at image 0, we have all boxes of ground truth class 0  
            num_gts = len(ground_truth_box)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_box): # 0, [img_0, class_0, prob, x1,y1,x2,y2]
                IOU = iou(torch.tensor(detection[3:]),
                          torch.tensor(gt[3:]), box_format) # calculate iou betweent pred box 1 to every other gt box
                
                if IOU > best_iou:
                    best_iou = IOU
                    best_gt_idx = idx # and in this way we can get the best fitted gt box of pred box
            
            if best_iou > iou_threshold: # when this is a solid prediction, say we have 0.8 iou > 0.5 iou_thred
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                                # current img # some gt box idx
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1 # means we have checked this gt box result
                else: 
                    FP[detection_idx] = 1 # if all boxes were checked, then the rest of pred box is over predicted, just FP
            else: # when this is not a solid prediction, it just a wrong predict, FP
                FP[detection_idx] = 1

        # for precision and recall
        # [1,1,0,1,0] -> [1,2,2,3,3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = torch.divide(TP_cumsum, (total_ture_boxes + epsilon))
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions)) # we want calculate start from 1, rather than 0
        recalls = torch.cat((torch.tensor([0]), recalls)) # x-axis

        average_precisions.append(torch.trapz(precisions, recalls)) # torch.trapz(y, x), get the aera under points
    
    return sum(average_precisions) / len(average_precisions)


def mAP_for_differ_iou_WIP(pred_boxes, true_boxes, range_list: list, num_classes=20):
    for i in range_list:
        results = []
        ap = mAP(pred_boxes, true_boxes, iou_threshold=i, box_format='corners', num_classes=num_classes)
        results.append(ap)

    return sum(results) / len(range_list)


def create_csv_file(image_dir: Path, save_path: str):
    """
    creat the index for example and label into a csv file
    in this case, we only put all image and label into a folder
    and do not need to create train folder with img, then another test folder with img
    """
    df = pd.DataFrame(columns=["images", "labels"])

    for _, _, filenames in os.walk(image_dir):
        for i, _ in enumerate(filenames):
            name = filenames[i].split(".")[0]
            label_name = f"{name}.txt"
            #break;
            df.loc[i] = [filenames[i], label_name]

        # df.__len__()
        df.to_csv(save_path, index=False)
        print("csv file has successfully created")
    
    return df


def create_train_test_csv_split(all_df: pd.DataFrame, num_sample, train_csv_path, test_csv_path):
    """
    split the image/label index into a csv file, rather than manipulate the image/label file 
    """
    train_df = all_df.sample(n=num_sample, replace=False)
    test_df = all_df[~all_df.isin(train_df)].dropna()

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print("train/test csv file has successfully created")
    print(f"train data: {len(train_df)}")
    print(f"test data: {len(test_df)}")

    return train_df, test_df


def convert_cellboxes(predictions, C, B=2, S=7): # for yolo
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.
    """

    # what's the prediction's shape?
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C+B*5)
    bboxes1 = predictions[..., C+1:C+5] # if class == 20, b1, b2 = [..., 21:25], [..., 26:30]
    bboxes2 = predictions[..., C+6:]

    scores = torch.cat( # [2, 32, 7, 7]
        (predictions[..., 3].unsqueeze(0), # [32, 7, 7] -> [1, 32, 7, 7]
         predictions[..., 8].unsqueeze(0)), dim=0)
    
    best_box = scores.argmax(0).unsqueeze(-1) # -> [32, 7, 7] -> [32, 7, 7, 1] # 0 for b1, 1 for b2
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2

    cell_indices = torch.arange(7).repeat(batch_size, S, 1).unsqueeze(-1)
    # [0, 1, 2, 3, 4, 5, 6] -> [[1-6],[1-6]...x7] -> [32, 7, 7, 1]
    
    # === so confused about this convert to global coord ===
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)

    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C+5]).unsqueeze(-1)

    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds


def cellboxes_to_boxes(out, C, S=7):
    # [32, 7, 7, 13]
    converted_pred = convert_cellboxes(out, C=C).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def get_bboxes(
    loader,
    model,
    num_classes,
    iou_threshold,
    prob_threshold,
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels, num_classes)
        bboxes = cellboxes_to_boxes(predictions, num_classes)

        for idx in range(batch_size):
            nms_boxes = nms(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=prob_threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx = train_idx+1

    model.train()
    return all_pred_boxes, all_true_boxes


def save_checkpoint(state_dict: dict, filename: str):
    """
        ckpt = {
                "state_dict": model.state_dict(),
                "optimizer": optima.state_dict()
            }
        save_checkpoint(ckpt, filename=xx.pth.tar)
    """
    print("=> Saving checkpoint")
    torch.save(state_dict, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


class En_De_cryption():
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.char = list('' + string.ascii_letters + string.digits + string.punctuation)
        self.key = self.char.copy()
        random.shuffle(self.key)


    def encryption(self, plain_text: str):
        cipher_text = ''

        for l in plain_text:
            index = self.char.index(l)
            cipher_text += self.key[index]

        return cipher_text


    def decryption(self, cipher_text: str):
        plain_text = ''

        for l in cipher_text:
            index = self.key.index(l)
            plain_text += self.char[index]

        return plain_text
    

def visualize_feat_map():
## this is an idea about how to visualize feature map:
## 1. get image and a model: 
            # model_path = "..."
            # img_idx = "a3"
            # image_path = f"E:\DeepLearning\\3_picked_demo_image\\{img_idx}.jpg"
            # model = torch.load(model_path)

            # all_child = list(model.children()) # len = 4 
            # type(all_child[0])
    
## 2. transform:
            # m_transform = mtransforms = transforms.Compose([
            #     transforms.Resize(size=(224, 224)),
            #     transforms.ToTensor()
            # ])

            # img = Image.open(image_path)
            # img = m_transform(img) # torch.Size([3, 224, 224])
            # img = np.asarray(img)
            # img = torch.from_numpy(img)
            # plt.imshow(img.permute(1,2,0))
            # img = img.unsqueeze(0) # torch.Size([1, 3, 224, 224])
            # img = img.to(device)
    
            # results = [all_child[0](img)]
            # for i in range(1, len(all_child)):
            #     results.append(all_child[i](results[-1]))
    
## 3. visualize:
            # plt.figure(figsize=(100, 50))
            # layer_viz = results[1].squeeze()
            # print("Stage: 2")
            # for i, f in enumerate(layer_viz):
            #     plt.subplot(4, 8, i + 1)
            #     plt.imshow(f.detach().cpu().numpy())
            #     plt.axis("off")
            # plt.show()
            # plt.close()
    return 
    
    
class RTG_Occuluder:
    """occulder = RTG_Occuluder(img_path=image_path, 
                                n_patches=n_patches, transforms=m_transform, device=device) # 8x8=64, patch size = 150x150
        img_container = occulder.generate_occlusion(mode="PIL")
        occulder.plot_occlusion(img_container)"""
    def __init__(self, img_path: str, n_patches: int, transforms, device) -> None:
        self.img_path = img_path
        self.device = device
        self.img = Image.open(self.img_path)
        self.n_patches = n_patches
        self.transforms = transforms
        self.occulusion_size = self.img.size[0] // self.n_patches # 120 


    def batch_single_img(self, img, show=False):
        img = self.transforms(img) # torch.Size([3, 224, 224])
        if show:
            plt.imshow(img.permute(1,2,0))
        img = img.unsqueeze(0) # torch.Size([1, 3, 224, 224])
        img = img.to(self.device)
        # print(f"the image has shape: {img.shape}")
        
        return img


    def generate_occlusion(self, mode, color: str="black"):
        if mode == "PIL":
            img_container = []
            for i in range(self.n_patches):
                for j in range(self.n_patches):
                    img = Image.open(self.img_path)
                    draw = ImageDraw.Draw(img)
                    draw.rectangle((j*self.occulusion_size, i*self.occulusion_size, 
                                    (j+1)*self.occulusion_size, (i+1)*self.occulusion_size), fill=color)
                    temp_img = self.batch_single_img(img)
                    img_container.append(temp_img)

            return img_container
        
        elif mode == "batched":
            img_container = []
            # img = Image.open(self.img_path)
            # img = self.batch_single_img(img)
            occulusion_size = img.shape[2] // self.n_patches
            for i in range(self.n_patches):
                for j in range(self.n_patches):
                    # img = Image.open(self.img_path)
                    img = self.batch_single_img(self.img)
                    img[0, :, i*self.occulusion_size:(i+1)*self.occulusion_size, 
                        j*self.occulusion_size:(j+1)*self.occulusion_size] = 0.
                    img_container.append(img)
            return img_container
    
    def plot_occlusion(self, img_container):
        plt.figure(figsize=(50, 50))
        for i, f in enumerate(img_container):
            plt.subplot(self.n_patches, self.n_patches, i+1)
            plt.imshow(img_container[i].squeeze(dim=0).detach().cpu().permute(1,2,0))
            plt.axis("off")
        plt.show()
        plt.close()

    def get_result(self, img_container, model, mode="logits"):
        rst_matrix = torch.randn(self.n_patches*self.n_patches).to(self.device)
        model.eval()
        for i, img in enumerate(img_container):
            with torch.inference_mode(): 
                if mode == "raw":                
                    rst = model(img)
                elif mode == "logits":
                    rst = torch.softmax(model(img), dim=1)
                else:
                    print("mode can only be 'raw' or 'logits'.")
            rst = rst.squeeze(0)
            rst_matrix[i] = rst[0]
        rst_matrix = rst_matrix.reshape((self.n_patches,self.n_patches)).cpu().numpy()

        return rst_matrix


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
# directory/file manipulate related funcs

def walk_through_dir(dir_path: pathlib.Path):
    """
    know about your dataset dir
    """

    for dirpath, dirname, filenames in os.walk(dir_path):
        print(f"There are {len(dirname)} directories and {len(filenames)} images in '{dirpath}'.")


def rename_get_rid_of_suffix(working_dir: str):

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


def move_img(image_dir_list: list[Path], aim_dir: Path):
    """move a list of image to a dirctory"""
    for img in image_dir_list:
        shutil.move(img, aim_dir)


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
def plot_trans(dataset_root_path: Path, 
               transform: torchvision.transforms, 
               n: int = 3, 
               seed=None):
    """
    select random img from a path list, and using transform, and visualize

    example:
        dataset_root_path: the root path of dataset, root/train/img...
        transform = transform.Compose([...])
    """

    if seed:
        random.seed(seed)

    img_path_list = list(dataset_root_path.glob('*/*/*.jpg'))
                         
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
    
    if type(results) != dict:
        results = pd.read_csv(results)
        results = results.iloc[:, 1:] # row, col
        results = results.to_dict("list")
    
    else:
        pass

    lr = results['learning rate']
    epochs = range(len(results['learning rate']))

    plt.figure(figsize=(7,7))
    plt.plot(epochs, lr, label='learning rate')
    plt.title('learning rate scheduler')
    plt.xlabel('Epochs')
    plt.legend()
    
    
def plot_loss_curves(results: Dict[str, List[float]]):
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


def plot_box_on_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    img_height, img_width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    
    # boxes: list[[b1...], [b2...]]
    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes: # [prob, xywh]
        box = box[2:]  
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle( # or maybe we need the bottom-left xy?? -- nope
            (upper_left_x * img_width, upper_left_y * img_height),
            box[2] * img_width, # box width
            box[3] * img_height, # box height
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def plot_yolo_prediciton(model, num_class, train_loader, device):
    for x, _ in train_loader:
        x = x.to(device)
        for i in range(8):
            bboxes = cellboxes_to_boxes(model(x), C=num_class)
            bboxes = nms(bboxes[i], iou_threshold=0.5, 
                            prob_threshold=0.4, box_format='midpoint')
            plot_box_on_image(x[i].permute(1,2,0).to('cpu'), bboxes)

        sys.exit()


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


class Dataset_from_CSV(Dataset):
    """csv file: ------ example ----- label -----
                    imgxxxxx.jpg        0 or imgxxxxx.txt
                    imgxxxxx.jpg        1
                        img..           ..
    """ 
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # .../data/imgxxxx.jpg
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0]) # row i, first col, the name of img
        img = io.imread(img_path)
        label = torch.tensor(int(self.annotations.iloc[index, 1])) # or maybe float?
        if self.transform:
            image = self.transform(img)
            
        return (image, label)


class SLIBR_Dataset(Dataset):
    def __init__(self, csv_file, img_dir, 
                 label_dir, C, S=7, B=2,
                 transform=None):
        super().__init__()
        
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S # 7x7
        self.B = B # bbox [p, xywh]
        self.C = C # class
        # final tensor is: [N, 7,7,13(3+5+5)] 
        # 0,1,0|p1, xywh|p2, xywh
        # 0,1,2,3,  4-7  8, 9-12

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        boxes = []

        with open(label_path) as f:
            for label in f.readlines(): # "0 0.479583 0.366250 0.210833 0.277500", it's a str
                class_label, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split() # ['0', '0.479583', '0.366250', '0.210833', '0.277500']
                ] # label is int, coord is float

                boxes.append([class_label, x, y, w, h])

        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        if self.transform:
            image, boxes = self.transform(image, boxes) # aug and resize should both happens at box and image
        
        label_matrix = torch.zeros((self.S, self.S, self.C+self.B*5)) # one bounding box per cell, and we just use one box

        for box in boxes: # let's assign those box to cells
            # convert every thing to fit label matrix
            class_label, x, y, w, h = box.tolist()
            class_label = int(class_label)
            # let's draw and figure out 
            # img label location reletive to cell        
            i, j = int(self.S * y), int(self.S * x) # the row and col of cell it belongs to, amazing
            x_cell, y_cell = self.S * x - j, self.S * y - i
            w_cell, h_cell = (self.S * w, self.S * h)

            if label_matrix[i, j, 3] == 0: # if this cell has not assigned for an object (initially all is 0)
                label_matrix[i, j, 3] = 1  # we sign a 100% prob, say this cell has obj
                box_coord = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_matrix[i, j, self.C+1:self.C+5] = box_coord
                label_matrix[i, j, class_label] = 1 # one-hot encode
        
        return image, label_matrix # label_matrix could be sparse, the rest of it is 0
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes
    

def create_dataloaders(
    train_dir: str, 
    valid_dir: str, 
    transform: transforms.Compose,
    batch_size: int, 
    test_transform: transforms.Compose = None, 
    num_workers: int = 0,
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

  if test_dir:
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


def lr_scheduler_setting(optima: torch.optim.Optimizer,
                        linearLR_factor: float = 0.1,
                        expLR_gamma: float = 0.95,
                        constLR_factor: float = 0.1,
                        mileston1: int = 30,
                        mileston2: int = 60,
                        epochs: int = 100):

        
    last = epochs-mileston2
    optima = optima


    if mileston1 > mileston2 or mileston1 > epochs:
        raise ValueError('mileston1 should smaller than epochs or mileston2')
    if mileston2 < mileston1 or mileston2 > epochs:
        raise ValueError('mileston2 should larger than mileston1 or smaller than epochs')

    scheduler1 = torch.optim.lr_scheduler.LinearLR(optima, start_factor=linearLR_factor)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optima, gamma=expLR_gamma)  # also need to tune gamma here
    scheduler3 = torch.optim.lr_scheduler.ConstantLR(optima, factor=constLR_factor, total_iters=last)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optima, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[mileston1, mileston2])

    return scheduler


def general_train_setup(Model: nn.Module,
                        train_path: Path, 
                        valid_path: Path, 
                        test_path: Path,
                        transform: transforms, 
                        test_transform: transforms,
                        batch_size: int = 8, 
                        num_worker: int = 8,  # cpu cores
                        init_lr: float = 0.01
                        ):
    
    """
    quick setup for a training, initially needs a test_dir
    
    Returns:
        a dict that contain dataloader, lr_scheduler(if needed), loss_fn, optimizing_func, classnames
    """

    loss_fn = torch.nn.CrossEntropyLoss()
    optima = torch.optim.AdamW(params=Model.parameters(), lr=init_lr, eps=1e-3) # 0.01


    if test_path:
        train_dataloader, valid_dataloader, test_dataloader, class_name = create_dataloaders(train_dir=train_path,
                                                                                        valid_dir=valid_path,
                                                                                        test_dir=test_path,
                                                                                        test_transform=test_transform,
                                                                                        batch_size=batch_size,
                                                                                        num_workers=num_worker,
                                                                                        transform=transform,
                                                                                        pin_mem=True)
        
        return {'train_dataloader': train_dataloader, 
                'valid_dataloader': valid_dataloader, 
                'test_dataloader': test_dataloader,
                'class_name': class_name, 
                'loss_fn': loss_fn, 
                'optima': optima}
    else:
        train_dataloader, valid_dataloader, class_name = create_dataloaders(train_dir=train_path,
                                                                            valid_dir=valid_path,
                                                                            test_dir=test_path,
                                                                            test_transform=test_transform,
                                                                            batch_size=batch_size,
                                                                            num_workers=num_worker,
                                                                            transform=transform,
                                                                            pin_mem=True)
        return {'train_dataloader': train_dataloader, 
                'valid_dataloader': valid_dataloader, 
                'class_name': class_name, 
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

        y_pred_t = Model(X) # i wonder if this place needs softmax to cal the loss? -- no need
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
            # pred_label = torch.argmax(raw_logits, dim=1)
            pred_label = raw_logits.argmax(dim=1)
            
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


def train_yolo(model, epochs, train_loader, optima, loss_fn, 
               num_class, iou_threshold, prob_threshold, device):
    
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for _ in range(epochs):
        pred_box, true_box = get_bboxes(train_loader, model, num_classes=num_class, 
                                        iou_threshold=iou_threshold, threshold=prob_threshold)
        maps = mAP(pred_box, true_box, num_classes=num_class, 
                   iou_threshold=iou_threshold, box_format="midpoint")
        print(f"train mAP: {maps}")       

        for _, (x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            mean_loss.append(loss.item())
            optima.zero_grad()
            loss.backward()
            optima.step()
            loop.set_postfix(loss = loss.item())

        mean_loss = sum(mean_loss)/len(mean_loss)
        print(f"mean loss: {mean_loss}")


class YoloLoss(nn.Module):
    def __init__(self, C, S=7, B=2):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C+self.B*5)
        # 7x7x30, 20 class + prob + 4 coord
        iou_b1 = iou(predictions[..., self.C+1:self.C+5], target[..., self.C+1:self.C+5]) # tensor([0.5533])
        iou_b2 = iou(predictions[..., self.C+6:], target[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # torch.Size([2, 1]), doing this is for the clarify to best box
        iou_max, best_box = torch.max(ious, dim=0) # max, argmax
        # if the second box is better, best_box = tensor([1]) ----- tensor(1) will have shape problem?
        exists_box = target[..., self.C].unsqueeze(3) # identity of obj i, 0 or 1, is there's an obj in cell i?
                    # batch?, 7x7x1, target[..., 20:21] is also the same?
        
        # ===================
        # for box coordinates
        # ===================
        box_pred_coords = exists_box * (
            (
                best_box * predictions[..., self.C+6:] \
                    + (1-best_box) * predictions[..., self.C+1:self.C+5] #torch.Size([batch, 7, 7, 4])
            )
        )
        
        box_target_coords = exists_box * target[..., self.C+1:self.C+5]
            # take sqrt for width and height 
        box_pred_coords[..., 2:4] = torch.sign(box_pred_coords[..., 2:4]) \
            * torch.sqrt(torch.abs(box_pred_coords[..., 2:4].clone())+1e-6) # .clone() is for debugging of inplace operation
            # 1. keep gradient direction correct, 2. avoid 0 and negative value
        box_target_coords[..., 2:4] = torch.sqrt(box_target_coords[..., 2:4])

        box_loss = self.mse( # N, S, S, 4 -> N*S*S, 4, same as dim=2
            torch.flatten(box_pred_coords, end_dim=-2),
            torch.flatten(box_target_coords, end_dim=-2)
        )

        # ===================
        # for obj loss
        # ===================
        obj_prob = ( # probabilty
            best_box * predictions[..., self.C+5:self.C+6] \
                + (1-best_box) * predictions[..., self.C:self.C+1]
        )
            # N*S*S*1
        obj_loss = self.mse(
            torch.flatten(exists_box * obj_prob),
            torch.flatten(exists_box * target[..., self.C:self.C+1])
        )

        # ===================
        # for no obj loss
        # ===================
        # N,S,S,1 -> N, S*S*1 # can we flatten to N*S*S*1?
        no_obj_loss = self.mse(
            torch.flatten((1-exists_box)*predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1-exists_box)*target[..., self.C:self.C+1], start_dim=1),
        )

        no_obj_loss = no_obj_loss + self.mse(
            torch.flatten((1-exists_box)*predictions[..., self.C+5:self.C+6], start_dim=1),
            torch.flatten((1-exists_box)*target[..., self.C:self.C+1], start_dim=1)
        )

        # ===================
        # for class loss
        # ===================
        # N*S*S, 20
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss \
            + obj_loss + self.lambda_noobj * no_obj_loss \
            + class_loss
        )

        return loss
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

    set_seeds()

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

# ————————————————————————————————————————————————————————————————————————————————————————————————————————————
## other tool/func/method...


def bin_search(nums:list, target) -> int:
    """search a target on a sorted list(upsenting)
        return the idx of target, other wise return -1
    """
    left = 0
    right = len(nums) - 1
    while left <= right:
        middle = (right + left) // 2
        if nums[middle] == target:
            return middle
        elif nums[middle] > target:
            right = middle - 1
        else:
            left = middle + 1
    return -1

# global min = f(512, 404.2319) = -959.6407, xi is at range [-512, 512]
def eggholder(x):
    f = -(x[1] + 47) * np.sin(np.sqrt(abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(
        np.sqrt(abs(x[0] - (x[1] + 47))))
    return f

def tabel_func(x):  
        return -np.abs(np.sin(x[0]) * np.cos(x[1]) *\
                        np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))

def waves(x):
    return np.sin(np.sqrt(x[0]**2+x[1]**2))/np.sqrt(x[0]**2+x[1]**2) + \
    np.exp((np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1])/2)) - 2.71289

# plot those func
# x_range = np.mgrid[-512:512:512j, -512:512:512j] # [start, end, step]
# ax = plt.subplot(111, projection="3d")
# ax.plot_surface(x_range[0], x_range[1], tabel_func(x_range), rstride=1, cstride=1, cmap=cm.jet)
# plt.show()

class Particle_for_PSO:
    def __init__(self, pos_constrain, v_constrain, func):
        # a particle only has two attri
        # 1. position, 2. velocity
        self.pos_constrain = pos_constrain
        self.v_constrain = v_constrain
        self.func = func

        self.pos = np.random.uniform(-pos_constrain, pos_constrain, 2) # initialize particle position
        self.v = np.random.uniform(-v_constrain, v_constrain, 2) # initialize particle velocity
        # just a random value between max and min
        # array([-5.12129409,  4.06149909])
        self.pbest_pos = self.pos # local best pos, initialize, same as pos
        self.pbest_value = func(self.pbest_pos) # local best value, initialize f(x)

    def get_pos(self):
        return self.pos
    def get_pb_pos(self):
        return self.pbest_pos
    def get_pb_value(self): 
        return self.pbest_value 
    def get_v(self):
        return self.v
    def set_pos(self, pos):
        self.pos = pos
    def set_v(self, v):
        self.v = v

    def set_pb_pos(self): # update local best pos
        new_pos_value = self.func(self.pos)
        if new_pos_value < self.pbest_value:
            self.pbest = self.pos
            self.pbest_value = new_pos_value
    
class PSO:
    """
    p = PSO(population=200, v_constrain=50.0, pos_constrain=512.0, c1=1.4995,
        c2=1.4995, epoch=50, func=eggholder)
    p.update()
    """
    def __init__(self, population,
                 v_constrain, pos_constrain, c1,
                 c2, epoch, func=eggholder) -> None:
        self.population = population
        self.func = func
        self.v_constrain = v_constrain
        self.pos_constrain = pos_constrain
        self.c1 = c1 # self aware, the larger the better at global searching
        self.c2 = c2 # social aware, the larger the easier fall into some local optim
        self.epoch = epoch
        self.w = np.flipud(np.linspace(.4, .9, epoch)) # using a linear weight
        # Reverse the order of elements along axis 0 (up/down)
        # 0.9, 0.89, 0.88, .... -> 0.4

        self.gb_pos = 0 # global best pos, initialize
        self.gb_value = 0
        self.gb_values = np.ones(epoch)

        # creat a lot of particle into a list [.....]
        self.particles = [Particle_for_PSO(pos_constrain, v_constrain, func) for i in range(population)]

    def evaluate(self, particles: Particle_for_PSO):
        for p in particles:
            p.set_pb_pos() # update local best pos

            if self.gb_value > p.get_pb_value(): # if global best is larger than some local
                self.gb_value = p.get_pb_value()
                self.gb_pos = p.get_pb_pos()

    def update_v(self, w, particles: Particle_for_PSO):
        for p in particles:
            # v = w*pv + c1*rand()*(p_best_pos-p_pos) + c2*rand()*(global_best_pos-p_pos)
            new_v = w * p.get_v() + self.c1 * np.random.random() * (p.get_pb_pos() - p.get_pos())\
                + self.c2 * np.random.random() * (self.gb_pos - p.get_pos())
            new_v[new_v > self.v_constrain] = self.v_constrain # if the new_v is too large, set to max value
            new_v[new_v < -self.v_constrain] = -self.v_constrain
            p.set_v(new_v)

    def update_pos(self, particles: Particle_for_PSO):
        for p in particles:
            new_pos = p.get_pos() + p.get_v() # vector adding 
            new_pos[new_pos > self.pos_constrain] = self.pos_constrain 
            new_pos[new_pos < -self.pos_constrain] = -self.pos_constrain
            p.set_pos(new_pos)

    def update(self):
        for i in range(self.epoch):
            self.evaluate(self.particles)
            self.update_v(self.w[i], self.particles,)
            self.update_pos(self.particles)
            self.gb_values[i] = self.gb_value

        plt.plot(range(self.epoch), self.gb_values)
        plt.show()

    def result(self):
        print(f"the Global value: {self.gb_values[self.epoch-1]}")
        print(f"the Global pos: {self.gb_pos}")
        return self.gb_values[self.epoch-1], self.gb_pos


def enclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class RTG_KNN:
    def __init__(self, K=3, ):
        self.K = K

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y

    def predict(self, X, type='clf'):
        preds = [self._predict(x, type) for x in X]
        return preds

    def _predict(self, x, type='clf'):
        # compute the distance
        distances = [enclidean_distance(x, x_train) for x_train in self.x_train]
        
        # get the closest k point
        k_idx = np.argsort(distances)[:self.K]
        k_labels = [self.y_train[i] for i in k_idx]
        
        if type == 'clf':
        # majority vote (for classification)
            most_common = Counter(k_labels).most_common()
            return most_common[0][0]

        elif type == 'reg':
        # average distance (for regression)
            avg_distance = np.average([distances[i] for i in k_idx])
            return avg_distance
        
    def acc(self, preds, labels):
        acc = 100 * np.sum(preds == labels) / len(labels)
        print(f'The accuracy is: {acc}%')
        return acc


class Node_for_decision_tree_version_A:
    def __init__(self, fearture=None, threshold=None, 
                 left=None, right=None, *,value=None) -> None:
        # *,value=None: if i want pass some value, i have to pass by name
        self.fearture = fearture
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class Desicion_tree_version_A:
    def __init__(self, min_samples_split=2, max_depth=100, n_feartures=None) -> None:
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_feartures=n_feartures 
        # might not use all the criteria at same time
        # it can be convient to build random forest
        self.root=None # keep track of previous node

    def fit(self, x, y):
        # in case that the num of feature is not exceed the actual fearture we have
        self.n_feartures = x.shape[1] if not self.n_feartures else \
            min(x.shape[1], self.n_feartures)
        self.root = self._grow_tree(x, y)

    def _grow_tree(self, x, y, depth=0):
        n_sample, n_feartures = x.shape
        n_label = len(np.unique(y))

        # check stopping criterial
        if (depth >= self.max_depth or 
            n_label==1 or 
            n_sample < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node_for_decision_tree_version_A(value=leaf_value)
        
        feat_idx = np.random.choice(n_feartures, self.n_feartures, replace=False)
        
        # find best split
        best_featrue, best_thresh = self._best_split(x, y, feat_idx)

        # create child node
        left_idx, right_idx = self._split(x[:, best_featrue], best_thresh)
        left = self._grow_tree(x[left_idx, :], y[left_idx], depth+1)
        right = self._grow_tree(x[right_idx, :], y[right_idx], depth+1)

        return Node_for_decision_tree_version_A(best_featrue, best_featrue, left, right)


    def _best_split(self, x, y, feat_idx):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idx:
            x_column = x[:, feat_idx]
            thresholds = np.unique(x_column)

            for thres in thresholds:
                # calculate IG
                IG = self._IG(y, x_column, thres)

                if IG > best_gain:
                    best_gain = IG
                    split_idx = feat_idx
                    split_threshold = thres

        return split_idx, split_threshold

    def _IG(self, y, x_col, thres):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idx, right_idx = self._split(x_col, thres)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # calculate weigthed avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate IG
        IG = parent_entropy - child_entropy

        return IG
    
    def _entropy(self, y):
        hist = np.bincount(y) # [1,2,3,1,2] -> [0,2,2,1] for the times that 0,1,2,3 appears
        p_x = hist / len(y) # [0, 2/5, 2/5, 1/5]

        return -np.sum([p*np.log(p) for p in p_x if p>0]) # in case log(0)=-inf 

    def _split(self, x_col, split_thres):
        left_idx = np.argwhere(x_col<=split_thres).flatten()
        right_idx = np.argwhere(x_col>split_thres).flatten()
        return left_idx, right_idx

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.fearture] <= node.threshold:
            return self._traverse_tree(x, node.left)
            
        return self._traverse_tree(x, node.right)

# data = datasets.load_breast_cancer()
# x, y = data.data, data.target
# X_train, X_test, y_train, y_test =train_test_split(x, y, test_size=0.2, random_state=42)
# clf = Desicion_tree_version_A()
# clf.fit(X_train, y_train)
# pred = clf.predict(X_test)
# acc = np.sum(y_test==pred) / len(y_test)
# print(acc)

class Node_for_desicion_tree_version_B():
    def __init__(self, feat_idx=None, threshold=None,
                 left=None, right=None, info_gain=None,
                 value=None):
        # the split condition is defined by feat_idx and threshold

        # for decision node
        self.feat_idx = feat_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value

class Decision_tree_version_B():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_feats = np.shape(X) # 150, 4

        # split until stopping
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find best split
            best_split = self.get_best_split(dataset, num_feats)

            # check if IG is positive
            if best_split["info_gain"]>0:
                # recur left and right
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)

                # return decision node
                return Node_for_desicion_tree_version_B(
                    best_split["feat_idx"], best_split["threshold"],
                    left_subtree, right_subtree, best_split["info_gain"]
                )

        # compute leaf node    
        leaf_value = self.calculate_leaf_value(y)
        return Node_for_desicion_tree_version_B(value=leaf_value)
    
    def get_best_split(self, dataset, num_feats):
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all features
        # i been thinking, how about we get a range, for every feature
        # that is the [min, max] at feature_i, and we search in this range
        # thres = np.arrange(min, max, step)
        # and we can adjust this step to see if we can get better?
        # ---- maybe not, since we are dealing with sparse data
        for feat_idx in range(num_feats):
            feat_values = dataset[:, feat_idx]
            # Find the unique elements of an array
            possible_thresholds = np.unique(feat_values)
            # loop over all the feature value present in the data

            for thres in possible_thresholds: 
                # get current split
                dataset_left, dataset_right = self.split(dataset, feat_idx, thres)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute IG
                    curr_info_gain = self.IG(y, left_y, right_y, "gini")

                    # update split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feat_idx"] = feat_idx
                        best_split["threshold"] = thres
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split
    
    def split(self, dataset, feat_idx, threshold):
        dataset_left = np.array([row for row in dataset if row[feat_idx] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feat_idx] > threshold])
        return dataset_left, dataset_right
    
    def IG(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == 'gini':
            gain = (self.gini_index(parent)-
                    (weight_l*self.gini_index(l_child) + 
                     weight_r*self.gini_index(r_child)))
        else: 
            gain = (self.entropy(parent)-
                    (weight_l*self.entropy(l_child) + 
                     weight_r*self.entropy(r_child)))
        return gain
    

    def entropy(self, y):
        class_label = np.unique(y) # 0,1,2..
        entropy = 0
        for cls in class_label:
            p_cls = len(y[y==cls]) / len(y)
            if p_cls > 0:
                entropy += -p_cls*np.log(p_cls)
        
        return entropy
            
    
    def gini_index(self, y):
        # hist = np.bincount(y)
        # p_x = hist / len(y)
        # return 1 - np.sum([p**2 for p in p_x])
        class_label = np.unique(y)
        gini = 0
        for cls in class_label:
            p_cls = len(y[y==cls]) / len(y)
            gini += p_cls**2
        
        return 1 - gini
    
    def calculate_leaf_value(self, y):
        # the value of leaf node is the class of majority
        y = list(y)
        return max(y, key=y.count)
    
    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feat_idx), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent+indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent+indent)

    def fit(self, x, y):
        dataset = np.concatenate((x,y), axis=1)
        self.root = self.build_tree(dataset)
            
    def predict(self, x):
        preds = [self.make_pred(x, self.root) for x in x]
        return preds
    
    def make_pred(self, x, tree):
        if tree.value != None: return tree.value
        feat_value = x[tree.feat_idx]
        if feat_value <= tree.threshold:
            return self.make_pred(x, tree.left)
        else:
            return self.make_pred(x, tree.right)

# clf = Decision_tree_version_B(min_samples_split=3, max_depth=3)
# clf.fit(X_train, y_train)
# clf.print_tree()
# y_pred = clf.predict(X_test)
# from sklearn.metrics import accuracy_score
# accuracy_score(y_test, y_pred)


# =================================
class Value:

    def __init__(self, data, _children=(), _op='', label='') -> None: 
        # _children is for the track of previous value
        # _op is for the track of previous operation, to know what operation created the value
        self.grad = 0.0
        self.data = data
        self._backward = lambda: None
        self._prev = set(_children) # turn a turple into a set, turple is faster than list
        self._op = _op
        self.label = label


    def __repr__(self) -> str: # __repr__ is similar to __str__, but __repr__ is use for inner debug, __str__ is use for show info to the user
        # representation
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # in case Value(a) + 1
        out = Value(self.data + other.data, (self, other), '+')
        def _backward(): # the "+=" is important, culmulate the gradient 
            self.grad += 1.0 * out.grad # noticed that if we have use a value more than once
            other.grad += 1.0 * out.grad # the derivative will not be correct

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # in case Value(a) * 5
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other): # other * self
        return self * other # 2 * a now also works
    
    def __radd__(self, other): # other + self
        return self + other # 2 + a now also works
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward

        return out

    # to achieve a/b: we can -> a * (1/b) -> a * b**-1
    # so is to achieve x**c
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        return out
    

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v): # DAG, topologist sort
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

class Neuron:
    """
    some features -> neuron ->
    x = [2., 3.]
    n = Neuron(2) #
    n(x)
    """
    def __init__(self, n_features):
        # maybe he_initialization?
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_features)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w*x + b
        # zip take two iter a,b
        # return another iter that contains tuple [(a[0], b[0]), (...)]
        raw_logits = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # start sum from self.b
        activated = raw_logits.tanh()
        return activated
    
    def parameters(self):
        return self.w + [self.b]
    
class NeuronLayer:
    """
    x = [2., 3.]
    n = NeuronLayer(2, 4)
    n(x)

    """
    def __init__(self, n_features, n_neurons):
        self.neurons = [Neuron(n_features) for _ in range(n_neurons)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()] # awsome
        # params = []
        # for n in self.neurons:
        #     ps = n.parameters()
        #     params.extend(ps)
        # return params
    
class RTG_MLP:
    """
    x = [2., 3., -1.]
    n = RTG_MLP(3, [4,4,1])
    n(x)
    """
    def __init__(self, n_features, n_hidden_units_per_layer:list):
        sz = [n_features] + n_hidden_units_per_layer
        self.layers = [NeuronLayer(sz[i], sz[i+1]) for i in range(len(n_hidden_units_per_layer))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

def RTG_MLP_train(mlp: RTG_MLP, epoch, lr, x_train, y_label):
    for e in range(epoch):
        # forward pass
        y_pred = [mlp(x) for x in x_train]
        # mse loss
        loss = sum((yout - ygt)**2 for ygt, yout in zip(y_label, y_pred))
        # BP
        for p in mlp.parameters(): # zero_grad before backward!
            p.grad = 0.0
        loss.backward()
        # update the grad to minimize the loss
        for p in mlp.parameters():
            p.data += -lr * p.grad 

        print(f"Epoch: {e} | loss: {loss.data}")