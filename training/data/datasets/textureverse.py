import glob
import logging
import os
import os.path as osp
import random

import cv2
import numpy as np

from data.base_dataset import BaseDataset


def _read_image_float(path, flags=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


class TextureVerseDataset(BaseDataset):
    def __init__(
        self,
        common_conf=None,
        common_config=None,
        split: str = "train",
        TextureVerse_DIR: str = "",
        min_num_images: int = 4,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        if common_conf is None:
            common_conf = common_config
        if common_conf is None:
            raise ValueError("TextureVerseDataset requires either common_conf or common_config")

        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.expand_ratio = expand_ratio
        self.TextureVerse_DIR = TextureVerse_DIR
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
            split_pattern = "selected_uids_for_training*"
        elif split == "test":
            self.len_train = len_test
            split_pattern = "selected_uids_for_eval*"
        else:
            raise ValueError(f"Invalid split: {split}")

        self.sequence_list = self._discover_sequences(split_pattern)
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No valid TextureVerse sequences found under {self.TextureVerse_DIR} for split={split}"
            )

        status = "Training" if self.training else "Testing"
        logging.info(f"TextureVerse_DIR is {self.TextureVerse_DIR}")
        logging.info(f"{status}: TextureVerse sequence count: {self.sequence_list_len}")
        logging.info(f"{status}: TextureVerse dataset length: {len(self)}")

    def _discover_sequences(self, split_pattern):
        sequences = []
        split_dirs = sorted(glob.glob(osp.join(self.TextureVerse_DIR, split_pattern)))
        for split_dir in split_dirs:
            for bucket in sorted(glob.glob(osp.join(split_dir, "*"))):
                if not osp.isdir(bucket):
                    continue
                for seq_dir in sorted(glob.glob(osp.join(bucket, "*"))):
                    if not osp.isdir(seq_dir):
                        continue
                    shaded_dir = osp.join(seq_dir, "shaded")
                    frame_paths = sorted(glob.glob(osp.join(shaded_dir, "*.png")))
                    if len(frame_paths) >= self.min_num_images:
                        sequences.append(seq_dir)
        return sequences

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_dir = self.sequence_list[seq_index % self.sequence_list_len]
        else:
            seq_dir = seq_name

        frame_paths = sorted(glob.glob(osp.join(seq_dir, "shaded", "*.png")))
        available_ids = np.arange(len(frame_paths))
        num_available_images = len(available_ids)
        if num_available_images < img_per_seq:
            raise FileNotFoundError(
                f"Sequence {seq_dir} only has {num_available_images} frames, but {img_per_seq} are required."
            )

        idx_ids = np.random.choice(
            num_available_images, img_per_seq, replace=self.allow_duplicate_img
        )
        if self.get_nearby:
            idx_ids = self.get_nearby_ids(
                idx_ids, num_available_images, expand_ratio=self.expand_ratio
            )
        ids = available_ids[idx_ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        original_sizes = []
        albedo_list = []
        metallic_list = []
        roughness_list = []
        normal_list = []
        shading_list = []
        mask_albedo_list = []
        mask_metallic_list = []
        mask_roughness_list = []
        mask_normal_list = []
        mask_shading_list = []

        for image_idx in ids:
            filename = osp.basename(frame_paths[image_idx])
            image_filepath = osp.join(seq_dir, "shaded", filename)
            albedo_filepath = osp.join(seq_dir, "basecolor", filename)
            metallic_filepath = osp.join(seq_dir, "metallic", filename)
            roughness_filepath = osp.join(seq_dir, "roughness", filename)
            normal_filepath = osp.join(seq_dir, "normal_png", filename)

            image = _read_image_float(image_filepath)
            albedo = _read_image_float(albedo_filepath)
            metallic = _read_image_float(metallic_filepath)[:, :, :1]
            roughness = _read_image_float(roughness_filepath)[:, :, :1]
            normal = _read_image_float(normal_filepath)
            normal = normal * 2.0 - 1.0
            eps = 1e-10
            normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + eps)
            shading = image.copy()

            original_size = np.array(image.shape[:2])

            invalid_mask = np.zeros(image.shape[:2], dtype=bool)
            mask_albedo = ~invalid_mask
            mask_albedo[(albedo < 0.004).all(axis=-1)] = False
            mask_metallic = ~invalid_mask
            mask_roughness = ~invalid_mask
            mask_normal = ~invalid_mask
            mask_shading = ~invalid_mask

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

            images.append(image)
            original_sizes.append(original_size)
            albedo_list.append(albedo)
            metallic_list.append(metallic)
            roughness_list.append(roughness)
            normal_list.append(normal)
            shading_list.append(shading)

            mask_albedo_list.append(mask_albedo)
            mask_metallic_list.append(mask_metallic)
            mask_roughness_list.append(mask_roughness)
            mask_normal_list.append(mask_normal)
            mask_shading_list.append(mask_shading)

        batch = {
            "seq_name": "textureverse_" + osp.basename(seq_dir),
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
