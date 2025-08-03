import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from grounding_sam2_utils.video_utils import create_video_from_images
from grounding_sam2_utils.common_utils import CommonUtils
from grounding_sam2_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy

def setup_environment():
    """Sets up the computing environment, especially GPU settings."""
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    return device

def load_models(device, model_name, grounding_model_id):
    """Loads the SAM2 and Grounding DINO models."""
    image_predictor = SAM2ImagePredictor.from_pretrained(model_name)
    video_predictor = SAM2VideoPredictor.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
    return video_predictor, image_predictor, processor, grounding_model

def prepare_directories_and_frames(video_dir, output_dir):
    """Creates output directories and returns sorted frame names."""
    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)
    CommonUtils.creat_dirs(result_dir)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split("_")[-1]))
    return mask_data_dir, json_data_dir, result_dir, frame_names

def save_frame_results(frame_idx, frame_masks_info, mask_data_dir, json_data_dir):
    """Saves the mask and JSON data for a single frame."""
    mask = frame_masks_info.labels
    mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
    for obj_id, obj_info in mask.items():
        mask_img[obj_info.mask == True] = obj_id

    mask_img_np = mask_img.cpu().numpy().astype(np.uint16)
    np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img_np)

    json_data = frame_masks_info.to_dict()
    json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
    with open(json_data_path, "w") as f:
        json.dump(json_data, f)


def process_frame_batch(start_frame_idx, frame_names, step, video_dir, text, device,
                          processor, grounding_model, image_predictor, video_predictor,
                          inference_state, global_masks, objects_count,
                          mask_data_dir, json_data_dir, prompt_type):
    """Processes a batch of frames: detection, SAM prediction, video propagation, saving results."""
    print(f"Processing frames starting from index: {start_frame_idx}")
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    image = Image.open(img_path)
    image_base_name = os.path.splitext(frame_names[start_frame_idx])[0]
    current_frame_masks = MaskDictionaryModel(promote_type=prompt_type, mask_name=f"mask_{image_base_name}.npy")

    # Run Grounding DINO
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.25, target_sizes=[image.size[::-1]]
    )

    input_boxes = results[0]["boxes"]
    labels = results[0]["labels"]

    if input_boxes.shape[0] > 0:
        # Run SAM Image Predictor
        image_predictor.set_image(np.array(image.convert("RGB")))
        masks, scores, logits = image_predictor.predict(
            point_coords=None, point_labels=None, box=input_boxes, multimask_output=False,
        )

        if masks.ndim == 2: # Handle single mask case
            masks = masks[None]
        elif masks.ndim == 4: # Handle batch dimension if present
             masks = masks.squeeze(1)

        # Register masks
        if current_frame_masks.promote_type == "mask":
            current_frame_masks.add_new_frame_annotation(
                mask_list=torch.tensor(masks).to(device),
                box_list=torch.tensor(input_boxes).to(device),
                label_list=labels
            )
        else:
            raise NotImplementedError("SAM 2 video predictor currently only supports mask prompts")

        # Update global mask dictionary
        objects_count = current_frame_masks.update_masks(
            tracking_annotation_dict=global_masks, iou_threshold=0.8, objects_count=objects_count
        )
        print(f"Updated objects count: {objects_count}")
    else:
        print(f"No objects detected in frame {frame_names[start_frame_idx]}, merging with previous masks.")
        current_frame_masks = copy.deepcopy(global_masks) # Use previous state if no new detections

    if len(current_frame_masks.labels) == 0:
        print(f"No objects to track from frame {start_frame_idx}, saving empty data for next {step} frames.")
        end_frame_idx = min(start_frame_idx + step, len(frame_names))
        current_frame_masks.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list=frame_names[start_frame_idx:end_frame_idx])
        # Return the potentially updated global_masks (even if empty) and objects_count
        return global_masks, objects_count
    else:
        # Propagate masks using Video Predictor
        video_predictor.reset_state(inference_state)
        for object_id, object_info in current_frame_masks.labels.items():
             _, _, _ = video_predictor.add_new_mask(
                inference_state, start_frame_idx, object_id, object_info.mask,
            )

        video_segments = {}
        last_propagated_masks = None
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):

            frame_masks = MaskDictionaryModel()
            if not out_obj_ids: # Skip if no objects tracked in this frame
                continue

            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0)
                class_name = current_frame_masks.get_target_class_name(out_obj_id)
                if class_name is None:
                    print(f"Warning: Could not find class name for object ID {out_obj_id} in frame {out_frame_idx}. Skipping.")
                    continue

                object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=class_name)
                object_info.update_box() # Calculate bounding box from mask
                frame_masks.labels[out_obj_id] = object_info

            if frame_masks.labels: # Only update if there are valid labels
                image_base_name = os.path.splitext(frame_names[out_frame_idx])[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                # Assuming all masks in a frame have the same shape
                first_mask = next(iter(frame_masks.labels.values())).mask
                frame_masks.mask_height = first_mask.shape[-2]
                frame_masks.mask_width = first_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                last_propagated_masks = copy.deepcopy(frame_masks) # Keep track of the last successfully propagated masks

        print(f"Propagated {len(video_segments)} frames.")

        # Save results for the propagated frames
        for frame_idx, frame_masks_info in video_segments.items():
            save_frame_results(frame_idx, frame_masks_info, mask_data_dir, json_data_dir)

        # Update global_masks with the last valid propagated masks for the next batch
        if last_propagated_masks:
             global_masks = last_propagated_masks
        else:
             # If propagation failed entirely for this batch, keep the masks from the start_frame_idx
             global_masks = current_frame_masks


        return global_masks, objects_count

def generate_visualization(video_dir, mask_data_dir, json_data_dir, output_dir, result_dir, output_video_path, frame_rate=15):
    """Draws masks and boxes on frames and saves the final annotated video."""
    print("Generating visualization...")
    CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, output_dir, result_dir)
    create_video_from_images(result_dir, output_video_path, frame_rate=frame_rate)
    print(f"Output video saved to: {output_video_path}")


def main(model_name='facebook/sam2-hiera-large', 
        grounding_model_id='IDEA-Research/grounding-dino-base', 
        text_prompt='car.', 
        video_dir='notebooks/videos/car', 
        output_dir='./outputs', 
        result_dir='./results',
        step=20, 
        prompt_type='mask'):

    # Setup
    device = setup_environment()
    video_predictor, image_predictor, processor, grounding_model = load_models(device, model_name, grounding_model_id)
    mask_data_dir, json_data_dir, vis_dir, frame_names = prepare_directories_and_frames(video_dir, output_dir)

    # Initialization for tracking
    inference_state = video_predictor.init_state(
        video_path=video_dir, 
        offload_video_to_cpu=True, 
        async_loading_frames=True
    )
    global_masks = MaskDictionaryModel(promote_type=prompt_type) # Holds masks from the last processed frame of the previous batch
    objects_count = 0 # Global counter for unique object IDs

    # Process video frames in batches
    print(f"Total frames to process: {len(frame_names)}")
    for start_frame_idx in range(0, len(frame_names), step):
        global_masks, objects_count = process_frame_batch(
            start_frame_idx, frame_names, step, video_dir, text_prompt, device,
            processor, grounding_model, image_predictor, video_predictor,
            inference_state, global_masks, objects_count,
            mask_data_dir, json_data_dir, prompt_type
        )

    # Generate final output video
    output_video_path = os.path.join(output_dir, "output.mp4")
    generate_visualization(video_dir, mask_data_dir, json_data_dir, vis_dir, result_dir, output_video_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate dynamic masks using GroundingSAM2')
    parser.add_argument('--model_name', type=str, default='facebook/sam2-hiera-large', help='Path to SAM2 model config')
    parser.add_argument('--grounding_model_id', type=str, default='IDEA-Research/grounding-dino-base', help='ID of the Grounding DINO model')
    parser.add_argument('--text_prompt', type=str, default='car.', help='Text prompt for Grounding DINO')
    parser.add_argument('--video_dir', type=str, default='', help='Path to video directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Path to output directory')
    parser.add_argument('--result_dir', type=str, default='./results', help='Path to result directory')
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        grounding_model_id=args.grounding_model_id,
        text_prompt=args.text_prompt,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        result_dir=args.result_dir
    )