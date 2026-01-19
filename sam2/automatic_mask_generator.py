# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.amg import (
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    MaskData,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SAM2AutomaticMaskGenerator:
    def __init__(
            self,
            model: SAM2Base,
            points_per_side: Optional[int] = 32, # Chinese: SAM沿着图像一边网格点（Prompt点）的数量，用来寻找物体。点越多，搜索越彻底，但速度越慢。（例如，64意味着在图像上放置64x64=4096个点）
                                                # English: How many grid points SAM should use along one side of the image to look for objects. More points = more thorough search, but slower. (e.g., 64 means 64x64=4096 points over the image).
            points_per_batch: int = 32,         # Chinese: SAM每次处理的这些网格点的数量。数字越大可能越快，但需要更多的计算机内存（GPU内存）。
                                                # English: How many of these grid points SAM processes at the same time. Higher number can be faster but needs more computer memory (GPU RAM).
            pred_iou_thresh: float = 0.8,       # Chinese: SAM认为一个Mask（分割掩码）质量有多好（0到1之间）。预测质量低于此数字的Mask会被丢弃。值越高=过滤越严格。
                                                # English: How good SAM thinks a mask is (0 to 1). Masks with a predicted quality below this number are thrown away. Higher = stricter filtering.
            stability_score_thresh: float = 0.95, # Chinese: 衡量Mask质量的另一个指标。它检查Mask边界在微小调整时变化了多少。低于此分数且过于“不稳定”（抖动）的Mask会被丢弃。值越高=过滤越严格。
                                                # English: Another measure of mask quality. It checks how much the mask changes if you slightly tweak its boundary. Masks that are too "wiggly" (unstable) below this score are thrown away. Higher = stricter filtering.
            stability_score_offset: float = 1.0, # Chinese: 计算“稳定性分数”时使用的一个技术设置。它会稍微调整Mask边界，以查看其稳定性。
                                                # English: A technical setting used when calculating the "stability score." It slightly adjusts the mask boundary to see how stable it is.
            mask_threshold: float = 0.0,        # Chinese: SAM内部用于将其原始Mask预测（连续值）转换为清晰的“开”或“关”（二值）Mask的截止值。默认0.0表示使用标准的0.5概率截止。
                                                # English: The internal cutoff value SAM uses to turn its raw mask prediction (continuous values) into a clear "on" or "off" (binary) mask. Default 0.0 means using the standard 0.5 probability cutoff.
            box_nms_thresh: float = 0.7,        # Chinese: 用于处理重叠Mask的“非极大值抑制”。如果两个Mask在很大程度上覆盖了同一个物体（它们的边界框重叠超过这个百分比），则只保留质量更好的那个。避免重复检测。
                                                # English: "Non-Maximum Suppression" for overlapping masks. If two masks largely cover the same object (their bounding boxes overlap by more than this percentage), keep only the better one. Avoids duplicate detections.
            crop_n_layers: int = 0,             # Chinese: SAM应该额外分析的图像放大区域（裁剪块）的“层数”。这有助于找到在完整图像上可能被忽略的小物体。（0=不裁剪，1=一层裁剪等）
                                                # English: How many "layers" of zoomed-in image sections (crops) SAM should also analyze. This helps find small objects that might be missed on the full image. (0 = no crops, 1 = one layer of crops, etc.).
            crop_nms_thresh: float = 0.7,       # Chinese: 类似于box_nms_thresh，但专门用于移除来自不同放大区域（裁剪块）的重复Mask。
                                                # English: Similar to box_nms_thresh, but specifically for removing duplicate masks that come from different zoomed-in sections (crops) of the image.
            crop_overlap_ratio: float = 512 / 1500, # Chinese: 图像放大区域（裁剪块）之间应该有多少重叠。重叠越多有助于确保物体不会正好在裁剪块边界处被切断。
                                                # English: How much the zoomed-in image sections (crops) should overlap with each other. More overlap helps ensure objects aren't cut exactly at a crop boundary.
            crop_n_points_downscale_factor: int = 1, # Chinese: 在分析放大区域（裁剪块）时，每个裁剪块中的网格点数量应该减少多少。（例如，2表示第一层裁剪块中的点数减半，第二层减至四分之一等）
                                                # English: When analyzing zoomed-in sections (crops), how much to reduce the number of grid points in each crop. (e.g., 2 means half the points in the first layer of crops, a quarter in the second, etc.).
            point_grids: Optional[List[np.ndarray]] = None, # Chinese: 你可以提供自己特定的点列表供SAM检查，而不是使用'points_per_side'来创建网格。（通常保留为None）
                                                # English: Instead of using 'points_per_side' to create a grid, you can provide your own specific list of points for SAM to check. (Usually left as None).
            min_mask_region_area: int = 0,      # Chinese: 创建Mask后，任何小于此像素面积的微小分离部分或Mask内部的孔洞都会被移除，以使其更整洁。（需要安装OpenCV）
                                                # English: After creating masks, any tiny detached bits or holes inside a mask smaller than this pixel area will be removed to clean it up. (Requires OpenCV installed).
            output_mode: str = "binary_mask",   # Chinese: 最终Mask的输出格式。“binary_mask”为每个Mask提供一个清晰的开/关图像。其他选项用于更紧凑的存储。
                                                # English: What format the final masks should be in. "binary_mask" gives a clear on/off image for each mask. Other options are for more compact storage.
            use_m2m: bool = False,              # Chinese: SAM是否应该尝试利用先前生成的Mask信息来改进其Mask。可以提高结果，但可能更慢。
                                                # English: Whether SAM should try to refine its masks by using previously generated mask information. Can improve results but might be slower.
            multimask_output: bool = True,      # Chinese: 对于每个网格点，SAM有时可能会为同一个物体预测几个略有不同的Mask。如果为True，它会考虑所有这些选项。
                                                # English: For each grid point, SAM can sometimes suggest a few slightly different masks for an object. If True, it considers all these options.
            **kwargs,                           # Chinese: 任何未在此处列出的、SAM2模型可能理解的其他高级设置。
                                                # English: Any other advanced settings not listed here that the SAM2 model might understand.
        ) -> None:
        """
        Using a SAM 2 model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM 2 with a HieraL backbone.

        Arguments:
          model (Sam): The SAM 2 model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          mask_threshold (float): Threshold for binarizing the mask logits
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          use_m2m (bool): Whether to add a one step refinement using previous mask predictions.
          multimask_output (bool): Whether to output multimask at each point of the grid.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils  # type: ignore  # noqa: F401
            except ImportError as e:
                print("Please install pycocotools")
                raise e

        self.predictor = SAM2ImagePredictor(
            model,
            max_hole_area=min_mask_region_area,
            max_sprinkle_area=min_mask_region_area,
        )
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.use_m2m = use_m2m
        self.multimask_output = multimask_output

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2AutomaticMaskGenerator":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2AutomaticMaskGenerator): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image)

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size, normalize=True
            )
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_predictor()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        normalize=False,
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        points = torch.as_tensor(
            points, dtype=torch.float32, device=self.predictor.device
        )
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        masks, iou_preds, low_res_masks = self.predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=points.repeat_interleave(masks.shape[1], dim=0),
            low_res_masks=low_res_masks.flatten(0, 1),
        )
        del masks

        if not self.use_m2m:
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # Calculate and filter by stability score
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            # One step refinement using previous mask predictions
            in_points = self.predictor._transforms.transform_coords(
                data["points"], normalize=normalize, orig_hw=im_size
            )
            labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, ious = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)

            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def refine_with_m2m(self, points, point_labels, low_res_masks, points_per_batch):
        new_masks = []
        new_iou_preds = []

        for cur_points, cur_point_labels, low_res_mask in batch_iterator(
            points_per_batch, points, point_labels, low_res_masks
        ):
            best_masks, best_iou_preds, _ = self.predictor._predict(
                cur_points[:, None, :],
                cur_point_labels[:, None],
                mask_input=low_res_mask[:, None, :],
                multimask_output=False,
                return_logits=True,
            )
            new_masks.append(best_masks)
            new_iou_preds.append(best_iou_preds)
        masks = torch.cat(new_masks, dim=0)
        return masks, torch.cat(new_iou_preds, dim=0)
