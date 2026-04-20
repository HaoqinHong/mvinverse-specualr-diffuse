import os.path as osp
import logging
import random
import glob
import re
import cv2
import numpy as np
import pyexr

from data.dataset_util import *
from data.base_dataset import BaseDataset

class InteriorverseDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        Interiorverse_DIR: str = "/home/data/wxz/projects/mvinverse/datasets/interiorverse",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the InteriorverseDataset

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            Interiorverse_DIR (str): Directory path to Interiorverse data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.expand_ratio = expand_ratio
        self.Interiorverse_DIR = Interiorverse_DIR
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"Interiorverse_DIR is {self.Interiorverse_DIR}")

        txt_path = osp.join(self.Interiorverse_DIR, f"{split}.txt")
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                sequence_list = [line.strip() for line in f.readlines()]
        else:
            # For InteriorVerse, train/test split should be given in the dataset instead of generate on your own
            raise ValueError(f"{self.Interiorverse_DIR} does not have {split}.txt split file")

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Interiorverse Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Interiorverse Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        # ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index % self.sequence_list_len]

        split_dir = osp.join(self.Interiorverse_DIR, "data")
        images_dir = osp.join(split_dir, seq_name)

        # select ids
        available_ids = np.array(sorted([int(os.path.basename(f).removesuffix("_im.exr")) for f in glob.glob(osp.join(images_dir, "*im.exr"))]))
        num_available_images = len(available_ids)
        # randomly select ids
        idx_ids = np.random.choice(num_available_images, img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            idx_ids = self.get_nearby_ids(idx_ids, num_available_images, expand_ratio=self.expand_ratio)
        ids = available_ids[idx_ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        original_sizes = []
        albedo_list = []
        metallic_list = []
        roughness_list = []
        normal_list =  []
        shading_list =  []
        mask_albedo_list = []
        mask_metallic_list = []
        mask_roughness_list = []
        mask_normal_list = []
        mask_shading_list = []

        for image_idx in ids:
            image_filepath = osp.join(images_dir, f"{image_idx:03d}_im.exr")
            albedo_filepath = osp.join(images_dir, f"{image_idx:03d}_albedo.exr")
            material_filepath = osp.join(images_dir, f"{image_idx:03d}_material.exr")
            normal_filepath = osp.join(images_dir, f"{image_idx:03d}_normal.exr")

            image = pyexr.open(image_filepath).get().clip(0, 1)
            albedo = pyexr.open(albedo_filepath).get().clip(0, 1)
            material = pyexr.open(material_filepath).get().clip(0, 1)
            # r channel for roughness, g channel for metallic
            metallic = material[:, :, 1:2]
            roughness = material[:, :, 0:1]
            normal = pyexr.open(normal_filepath).get().clip(-1, 1)
            # normalize
            eps = 1e-10
            normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + eps)
            # NOTE: dummy shading
            shading = np.ones_like(image, dtype=image.dtype)

            H, W, _ = image.shape
            invalid_mask = np.zeros((H, W), dtype=bool)

            mask_albedo = ~invalid_mask
            ### mask too dark or too bright albedo
            mask_albedo[(albedo < 0.004).all(axis=-1)] = False
            mask_metallic = ~invalid_mask
            mask_roughness = ~invalid_mask
            # mask_normal = ~invalid_mask
            ## NOTE: abandon its normal due to over exposure
            mask_normal = np.zeros_like(mask_albedo, dtype=bool)
            mask_shading = np.zeros_like(mask_albedo, dtype=bool)

            if albedo is None or image is None or metallic is None or roughness is None or normal is None: # 没有对应的albedo
                raise FileNotFoundError

            assert image.shape[:2] == albedo.shape[:2], f"Image and albedo shape mismatch: {image.shape[:2]} vs {albedo.shape[:2]}"

            original_size = np.array(image.shape[:2])

            (
                image,
                albedo,
                metallic,
                roughness,
                _,
                _,
                _,
                normal,
                shading,
                mask_albedo,
                mask_metallic,
                mask_roughness,
                _,
                _,
                _,
                mask_normal,
                mask_shading,
            ) = self.process_one_image_wo_geo(
                image,
                albedo=albedo,
                metallic=metallic,
                roughness=roughness,
                normal=normal,
                shading=shading,
                mask_albedo=mask_albedo,
                mask_metallic=mask_metallic,
                mask_roughness=mask_roughness,
                mask_normal=mask_normal,
                mask_shading=mask_shading,
                target_image_shape=target_image_shape,
            )

            for name, arr in {
                "image": image,
                "albedo": albedo,
                "metallic": metallic,
                "roughness": roughness,
                "normal": normal,
                "shading": shading,
                "mask_albedo": mask_albedo,
                "mask_metallic": mask_metallic,
                "mask_roughness": mask_roughness,
                "mask_normal": mask_normal,
                "mask_shading":  mask_shading,
            }.items():
                if (arr.shape[:2] != target_image_shape).any():
                    logging.error(f"Wrong shape for {seq_name} ({name}): expected {target_image_shape}, got {arr.shape[:2]}")
                    continue

            images.append(image)
            original_sizes.append(original_size)
            albedo_list.append(albedo)
            metallic_list.append(metallic)
            roughness_list.append(roughness)
            normal_list.append(normal)
            shading_list.append(shading)

            mask_albedo_list.append(mask_albedo)
            mask_roughness_list.append(mask_roughness)
            mask_metallic_list.append(mask_metallic)
            mask_normal_list.append(mask_normal)
            mask_shading_list.append(mask_shading)

        set_name = "interiorverse"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(images),
            "images": images,
            "original_sizes": original_sizes,
            "albedo": albedo_list,
            "metallic": metallic_list,
            "roughness": roughness_list,
            "normal": normal_list,
            "shading": shading_list,
            "mask_albedo": mask_albedo_list,
            "mask_metallic": mask_metallic_list,
            "mask_roughness": mask_roughness_list,
            "mask_normal": mask_normal_list,
            "mask_shading": mask_shading_list,
        }
        return batch
