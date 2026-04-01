import os.path as osp
import logging
import random
import glob
import re
import cv2
import numpy as np
import h5py 

from data.dataset_util import *
from data.base_dataset import BaseDataset

from pi3.utils.color import srgb_to_linear

def read_hdf(path):
    fd = h5py.File(path, 'r')
    data = np.array(fd['dataset'])
    return data

class HypersimDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        Hypersim_DIR: str = "/home/data/wxz/projects/mvinverse/datasets/hypersim",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the HypersimDataset

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            Hypersim_DIR (str): Directory path to Hypersim data.
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
        self.Hypersim_DIR = Hypersim_DIR
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"Hypersim_DIR is {self.Hypersim_DIR}")

        # Load or generate sequence list
        # TODO: Hypersim has no sequence list
        txt_path = osp.join(self.Hypersim_DIR, f"sequence_list_{split}.txt")
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                sequence_list = [line.strip() for line in f.readlines()]
        else:
            # Generate sequence list and save to txt            
            sequence_list = glob.glob(osp.join(self.Hypersim_DIR, f"{split}/*/_detail/cam_*"))
            sequence_list = [file_path.split('/')[-3] + "/" + file_path.split('/')[-1] for file_path in sequence_list]
            sequence_list = sorted(sequence_list)

            # Save to txt file
            with open(txt_path, 'w') as f:
                f.write('\n'.join(sequence_list))

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Hypersim Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Hypersim Data dataset length: {len(self)}")

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
        
        scene_name, cam_name = seq_name.split("/")

        split_dir = osp.join(self.Hypersim_DIR, "train") if self.training else osp.join(self.Hypersim_DIR, "test")
        images_dir = osp.join(split_dir, scene_name, "images", f"scene_{cam_name}_final_tonemap")

        # select ids
        available_ids = np.array(sorted([int(f.split(".")[1]) for f in glob.glob(osp.join(images_dir, "*.jpg"))]))
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
            image_filepath = osp.join(split_dir, scene_name, "images", f"scene_{cam_name}_final_tonemap", f"frame.{image_idx:04d}.tonemap.jpg")
            albedo_filepath = osp.join(split_dir, scene_name, "images", f"scene_{cam_name}_final_tonemap_albedo", f"frame.{image_idx:04d}.tonemap.jpg")
            # depth_filepath = osp.join(split_dir, scene_name, "images", f"scene_{cam_name}_geometry_hdf5", f"frame.{image_idx:04d}.depth_meters.hdf5")
            normal_filepath = osp.join(split_dir, scene_name, "images", f"scene_{cam_name}_geometry_hdf5", f"frame.{image_idx:04d}.normal_cam.hdf5")
            shading_filepath = osp.join(split_dir, scene_name, "images", f"scene_{cam_name}_final_tonemap_shading", f"frame.{image_idx:04d}.tonemap.jpg")

            image = read_image_cv2(image_filepath)
            albedo = read_image_cv2(albedo_filepath)  # TODO: 暂时不做gamma
            normal = read_hdf(normal_filepath)
            shading = read_image_cv2(shading_filepath)

            # if albedo is None or image is None or normal is None: 
            if albedo is None or image is None or shading is None: 
                raise FileNotFoundError

            # normalize image and albedo
            image = image / 255.0
            albedo = albedo / 255.0
            shading = shading / 255.0
            # srgb to linear
            image = srgb_to_linear(image)
            albedo = srgb_to_linear(albedo)
            shading = srgb_to_linear(shading)

            # creat dummy metallic, roughness
            H, W, _ = image.shape
            metallic = np.ones((H, W, 1), dtype=image.dtype)
            roughness = np.ones((H, W, 1), dtype=image.dtype)

            # create masks
            ## for albedo and normal, ones
            mask_shading = ~np.any(shading < 0.004, axis=2)
            mask_albedo = mask_shading.copy()
            mask_albedo[(albedo < 0.01).all(axis=-1)] = False
            # mask_albedo = np.ones((H, W), dtype=bool)
            # ### mask too dark or too bright albedo
            mask_normal = np.ones((H, W), dtype=bool)
            ## for metallic and roughness, zeros
            mask_metallic = np.zeros((H, W), dtype=bool)
            mask_roughness = np.zeros((H, W), dtype=bool)

            assert image.shape[:2] == albedo.shape[:2], f"Image and albedo shape mismatch: {image.shape[:2]} vs {albedo.shape[:2]}"
            assert image.shape[:2] == normal.shape[:2], f"Image and normal shape mismatch: {image.shape[:2]} vs {albedo.shape[:2]}"

            original_size = np.array(image.shape[:2])

            (
                image,
                albedo,
                metallic,
                roughness,
                normal,
                shading,
                mask_albedo,
                mask_metallic,
                mask_roughness,
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


            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
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

        set_name = "hypersim"
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
