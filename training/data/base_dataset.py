# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
from PIL import Image, ImageFile

from torch.utils.data import Dataset
from .dataset_util import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    """
    Base dataset class for VGGT and VGGSfM training.

    This abstract class handles common operations like image resizing,
    augmentation, and coordinate transformations. Concrete dataset
    implementations should inherit from this class.

    Attributes:
        img_size: Target image size (typically the width)
        patch_size: Size of patches for vit
        augs.scales: Scale range for data augmentation [min, max]
        rescale: Whether to rescale images
        rescale_aug: Whether to apply augmentation during rescaling
        landscape_check: Whether to handle landscape vs portrait orientation
    """
    def __init__(
        self,
        common_conf,
    ):
        """
        Initialize the base dataset with common configuration.

        Args:
            common_conf: Configuration object with the following properties, shared by all datasets:
                - img_size: Default is 518
                - patch_size: Default is 14
                - augs.scales: Default is [0.8, 1.2]
                - rescale: Default is True
                - rescale_aug: Default is True
                - landscape_check: Default is True
        """
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = common_conf.augs.scales
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.landscape_check = common_conf.landscape_check

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        """
        Get an item from the dataset.

        Args:
            idx_N: Tuple containing (seq_index, img_per_seq, aspect_ratio)

        Returns:
            Dataset item as returned by get_data()
        """
        while True:
            try:
                seq_index, img_per_seq, aspect_ratio = idx_N
                data = self.get_data(
                    seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
                )
                return data
            except FileNotFoundError as e:
                print(f"[Warning...] Error in processing seq_index {seq_index}..., {e}")
                new_seq_index = (seq_index + 1) % len(self)
                idx_N = (new_seq_index, img_per_seq, aspect_ratio)

        # seq_index, img_per_seq, aspect_ratio = idx_N
        # data = self.get_data(
        #     seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
        # )
        # return data

    def get_data(self, seq_index=None, seq_name=None, aspect_ratio=1.0):
        """
        Abstract method to retrieve data for a given sequence.

        Args:
            seq_index (int, optional): Index of the sequence
            seq_name (str, optional): Name of the sequence
            ids (list, optional): List of frame IDs
            aspect_ratio (float, optional): Target aspect ratio.

        Returns:
            Dataset-specific data

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass, i.e., each dataset should implement its own get_data method."
        )

    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.

        Args:
            aspect_ratio: Target aspect ratio

        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size

        # ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        image_shape = np.array([short_size, self.img_size])
        return image_shape

    def process_one_image_wo_geo(
        self,
        image,
        target_image_shape,
        albedo=None,
        metallic=None,
        roughness=None,
        normal=None,
        shading=None,
        mask_albedo=None,
        mask_metallic=None,
        mask_roughness=None,
        mask_normal=None,
        mask_shading=None,
    ):
        """
        - Center Crop if input size larger than target
        - Center Padding if input size smaller than target

        Args:
            image (np.ndarray): (height, width, channels)。
            target_image_shape (tuple): Desired output shape (target_height, target_width)。
            albedo (np.ndarray, optional):  Defaults to None.
            metallic (np.ndarray, optional):  Defaults to None.
            roughness (np.ndarray, optional):  Defaults to None.
            normal (np.ndarray, optional):  Defaults to None.
            mask_albedo (np.ndarray, optional):  Defaults to None.
            mask_metallic (np.ndarray, optional):  Defaults to None.
            mask_roughness (np.ndarray, optional):  Defaults to None.
            mask_normal (np.ndarray, optional):  Defaults to None.

        Returns:
            tuple: all materials and masks
        """
        def _crop_or_pad_image(image, target_shape):
            """
                image: (H, W, C) or (H, W)
            """
            if image is None:
                return None

            target_height, target_width = target_shape
            original_height, original_width = image.shape[:2]
            
            # --- Processing Height ---
            if original_height > target_height:
                # center crop
                start_y = (original_height - target_height) // 2
                image = image[start_y:start_y + target_height, :]
            elif original_height < target_height:
                # center padding
                pad_top = (target_height - original_height) // 2
                pad_bottom = target_height - original_height - pad_top
                if image.ndim == 3:
                    pad_width = ((pad_top, pad_bottom), (0, 0), (0, 0))
                elif image.ndim == 2:
                    pad_width = ((pad_top, pad_bottom), (0, 0))
                else:
                    raise RuntimeError(f"image dimension should be 2 or 3")
                image = np.pad(image, pad_width, mode='constant', constant_values=0)

            # --- Process Width ---
            current_width = image.shape[1]
            if current_width > target_width:
                # Center Crop
                start_x = (current_width - target_width) // 2
                image = image[:, start_x:start_x + target_width]
            elif current_width < target_width:
                # Center Padding
                pad_left = (target_width - current_width) // 2
                pad_right = target_width - current_width - pad_left
                if image.ndim == 3:
                    pad_width = ((0, 0), (pad_left, pad_right), (0, 0))
                elif image.ndim == 2:
                    pad_width = ((0, 0), (pad_left, pad_right))
                else:
                    raise RuntimeError(f"image dimension should be 2 or 3")
                image = np.pad(image, pad_width, mode='constant', constant_values=0)
                
            return image

        # all maps to process
        maps_to_process = [
            image, albedo, metallic, roughness, normal, shading,
            mask_albedo, mask_metallic, mask_roughness, mask_normal, mask_shading,
        ]
        
        processed_maps = [_crop_or_pad_image(m, target_image_shape) for m in maps_to_process]
        
        return tuple(processed_maps)

    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None):
        """
        TODO: add the function to sample the ids by pose similarity ranking.

        Sample a set of IDs from a sequence close to a given start index.

        You can specify the range either as a ratio of the number of input IDs
        or as a fixed integer window.


        Args:
            ids (list): Initial list of IDs. The first element is used as the anchor.
            full_seq_num (int): Total number of items in the full sequence.
            expand_ratio (float, optional): Factor by which the number of IDs expands
                around the start index. Default is 2.0 if neither expand_ratio nor
                expand_range is provided.
            expand_range (int, optional): Fixed number of items to expand around the
                start index. If provided, expand_ratio is ignored.

        Returns:
            numpy.ndarray: Array of sampled IDs, with the first element being the
                original start index.

        Examples:
            # Using expand_ratio (default behavior)
            # If ids=[100,101,102] and full_seq_num=200, with expand_ratio=2.0,
            # expand_range = int(3 * 2.0) = 6, so IDs sampled from [94...106] (if boundaries allow).

            # Using expand_range directly
            # If ids=[100,101,102] and full_seq_num=200, with expand_range=10,
            # IDs are sampled from [90...110] (if boundaries allow).

        Raises:
            ValueError: If no IDs are provided.
        """
        if len(ids) == 0:
            raise ValueError("No IDs provided.")

        if expand_range is None and expand_ratio is None:
            expand_ratio = 2.0  # Default behavior

        total_ids = len(ids)
        start_idx = ids[0]

        # Determine the actual expand_range
        if expand_range is None:
            # Use ratio to determine range
            expand_range = int(total_ids * expand_ratio)

        # Calculate valid boundaries
        low_bound = max(0, start_idx - expand_range)
        high_bound = min(full_seq_num, start_idx + expand_range)

        # Create the valid range of indices
        valid_range = np.arange(low_bound, high_bound)

        # Sample 'total_ids - 1' items, because we already have the start_idx
        sampled_ids = np.random.choice(
            valid_range,
            size=(total_ids - 1),
            replace=True,   # we accept the situation that some sampled ids are the same
        )

        # Insert the start_idx at the beginning
        result_ids = np.insert(sampled_ids, 0, start_idx)

        return result_ids
