"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import tqdm
import pathlib
import cv2

import numpy as np
from PIL import Image
import pdb

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

def save_img():
    save_root = "/labs3/amartel_data3/tingxiao/hover_net/kumar-mask/"
    open_root = "/labs3/amartel_data3/tingxiao/hover_net/kumar-patches/train/540x540_164x164/"
    file_list = os.listdir(open_root)
    file_list.sort()
    for file in file_list:
        base_name = file.split('.')[0]
        im_ann = np.load(open_root+file)
        #img = im_ann[:,:,0:3]
        #im = Image.fromarray(img.astype(np.uint8)).convert('RGB')
        #im.save("{0}/{1}.png".format(save_root, base_name))
        mask = im_ann[:,:,3:4]
        mask[mask>0]=1

        _,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[1]
        res = cv2.drawContours(mask, [cnt], 0, (0, 0, 255), 1)

        new_mask = Image.fromarray((res).astype(np.uint8)).convert('RGB')
        new_mask.save("{0}/{1}.png".format(save_root, base_name))
        '''
        mask = np.squeeze(mask, axis=2)
        mask = np.stack((mask,)*3, axis=-1)
        new_mask = Image.fromarray((mask*255).astype(np.uint8)).convert('RGB')
        new_mask.save("{0}/{1}.png".format(save_root, base_name))
        '''





def extract_img():

    type_classification = True

    win_size = [540, 540]
    step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "kumar"
    #save_root = "/labs3/amartel_data3/tingxiao/hover_net/%s-patches/" % dataset_name
    save_root = "/labs3/amartel_data3/tingxiao/hover_net/IHC-patches/" 

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (".tif", "/labs3/amartel_data3/tingxiao/hover_net/IHC/"),
            #"ann": (".mat", "/labs3/amartel_data3/tingxiao/hover_net/kumar/train/Labels/"),
        },
        #"valid": {
        #    "img": (".png", "/labs3/amartel_data3/tingxiao/hover_net/kumar/test_same/stain_images1/no_backgrounds/"),
        #    "ann": (".mat", "/labs3/amartel_data3/tingxiao/hover_net/kumar/test_same/Labels/"),
        #},
    }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        #ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%dx%d_%dx%d/" % (
            save_root,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        file_list = glob.glob(patterning("%s/*%s" % (img_dir, img_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem
            print(img_dir,base_name,img_ext)
            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            #ann = parser.load_ann("%s/%s%s" % (ann_dir, base_name.split('_')[0], ann_ext), type_classification,flag)
            #pdb.set_trace()
            # *
            #img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )
            for idx, patch in enumerate(sub_patches):
                im = Image.fromarray(patch.astype(np.uint8)).convert('RGB')
                im.save("{0}/{1}_{2:03d}.png".format(out_dir, base_name, idx))
                #np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()

def extract_new():

    type_classification = False

    win_size = [540, 540]
    step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "kumar"
    save_root = "/labs3/amartel_data3/tingxiao/hover_net/B-patches/"# % dataset_name

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (".png", "/labs3/amartel_data3/tingxiao/hover_net/kumar/train/B/"),
            "ann": (".mat", "/labs3/amartel_data3/tingxiao/hover_net/kumar/train/Labels/"),
        },
        "valid": {
            "img": (".png", "/labs3/amartel_data3/tingxiao/hover_net/kumar/train/B/"),
            "ann": (".mat", "/labs3/amartel_data3/tingxiao/hover_net/kumar/train/Labels/"),
        },
    }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%dx%d_%dx%d/" % (
            save_root,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        file_list = glob.glob(patterning("%s/*%s" % (img_dir, img_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem
            print(img_dir,base_name,img_ext)
            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            if 'blue' in base_name:
                flag = 'blue'
            if 'brown' in base_name:
                flag = 'brown'
            ann = parser.load_ann("%s/%s%s" % (ann_dir, base_name.split('_')[0], ann_ext), type_classification)
            #pdb.set_trace()
            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()


def extract_four_types():
    type_classification = True

    win_size = [540, 540]
    step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use aml
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "aml"#"kumar"
    save_root = "/labs3/amartel_data3/tingxiao/hover_net/%s-intens-patches/" % dataset_name
    dataset_info = {
        "train": {
            "img": (".png", "/labs3/amartel_data3/tingxiao/hover_net/AML-ROIS-intens/train/Images/"),
            "ann": (".mat", "/labs3/amartel_data3/tingxiao/hover_net/AML-ROIS-intens/train/Anno/"),
        },
        "valid": {
            "img": (".png", "/labs3/amartel_data3/tingxiao/hover_net/AML-ROIS-intens/valid/Images/"),
            "ann": (".mat", "/labs3/amartel_data3/tingxiao/hover_net/AML-ROIS-intens/valid/Anno/"),
        },
    }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%dx%d_%dx%d/" % (
            save_root,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem
            print(img_dir,base_name,img_ext)
            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification,5
            )

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *
            pbarx.update()
        pbarx.close()

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    extract_four_types()
    #extract_img()
    #save_img()
    #extract_new()
