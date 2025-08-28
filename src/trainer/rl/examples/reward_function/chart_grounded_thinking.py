import json
import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer
from typing import Dict
from examples.reward_function.chart_multibbox import calculate_bbox_iou
from examples.reward_function.chart_bbox import format_reward, accuracy_reward_with_error, extract_bbox_from_response

# def format_reward(response: str) -> float:
#     pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
#     format_match = re.fullmatch(pattern, response)
#     return 1.0 if format_match else 0.0




def compute_score_with_error(predict_str: str, ground_truth: str, extra_info, format_weight: float = 0.1) -> list[dict[str, float]]:
    # if not isinstance(reward_inputs, list):
    #     raise ValueError("Please use `reward_type=batch` for math reward function.")
    # breakpoint()
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)  # handle qwen2.5vl-32b format
    format_score = format_reward(predict_str)
    # coordinate_score = coordinate_reward(predict_str)
    if len(ground_truth) == 0:
        breakpoint()
    accuracy_score = accuracy_reward_with_error(predict_str, ground_truth, 
    extra_info["ground_truth_precision"], extra_info["error_margin"])
    # iou_score = 0
    pred_bboxes = extract_bbox_from_response(predict_str)
    coordinate_score = len(pred_bboxes) >= 1 
    # if pred_bboxes:
    #     iou_score = calculate_bbox_iou(pred_bboxes, gt_bbox=extra_info["bbox"])
    return {
        "overall": 0.9 * accuracy_score + 0.1 * (format_score * coordinate_score),
        "format": format_score,
        "coordinate": coordinate_score,
        # "iou": iou_score,
        "accuracy": accuracy_score,
    }


# def compute_score_with_bbox_error(predict_str: str, ground_truth: str, segmentation_mask=None, bbox=None) -> Dict[str, float]:
#     """
#     Compute medical scoring including standard score, bounding box IoU, and format score.

#     Args:
#         predict_str: The model's prediction string
#         ground_truth: The ground truth string
#         segmentation_mask: Ground truth segmentation mask tensor
#         bbox: Ground truth bounding box

#     Returns:
#         Tuple of (standard_score, bbox_score)
#         Note: bbox_score is a combination of IoU score and format score
#     """
#     # Calculate standard score
#     answer = extract_boxed_content(predict_str)
#     if answer == "None":
#         standard_score = 0.0  # no answer
#     else:
#         # Parse both prediction and ground truth into sets of conditions
#         predicted_conditions = parse_conditions(answer)
#         ground_truth_conditions = parse_conditions(ground_truth)

#         # Calculate true positives, false positives, and false negatives
#         true_positives = len(predicted_conditions.intersection(ground_truth_conditions))
#         false_positives = len(predicted_conditions - ground_truth_conditions)
#         false_negatives = len(ground_truth_conditions - predicted_conditions)

#         # Calculate F1 score components
#         precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
#         recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

#         # Calculate F1 score (harmonic mean of precision and recall)
#         standard_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     # Calculate format score (how well the JSON follows the expected format)
#     format_score = evaluate_bbox_format(predict_str)

#     # length score
#     if len(predict_str) > 600:  # ~200 words
#         length_score = 1
#     else:
#         length_score = len(predict_str) * 0.001


#     # Calculate bounding box IoU score
#     iou_score = 0.0
#     # Extract predicted bounding boxes from the response
#     json_data = extract_json_from_response(predict_str)
#     if json_data:
#         # Extract bounding boxes from the JSON
#         try:
#             pred_bboxes = []
#             if isinstance(json_data, list):
#                 for item in json_data:
#                     if isinstance(item, dict) and "bbox_2d" in item:
#                         pred_bboxes.append(item["bbox_2d"])
#             elif isinstance(json_data, dict) and "bbox_2d" in json_data:
#                 pred_bboxes.append(json_data["bbox_2d"])
#             elif isinstance(json_data, dict) and 'objects_of_interest' in json_data:
#                 for item in json_data['objects_of_interest']:
#                     if isinstance(item, dict) and "bbox_2d" in item:
#                         pred_bboxes.append(item["bbox_2d"])
#             # else:
#             #     print("Error: Invalid JSON format")
#             if random.random() < 0.0005:  # print every 0.5%
#                 print("[Bounding Box] ", json_data)
#                 print("[Formatted Bounding Box] ", pred_bboxes)
#                 print('[GT Bounding Box] ', bbox)

#             # Calculate IoU between predicted boxes and ground truth
#             if pred_bboxes:
#                 iou_score = calculate_bbox_iou(pred_bboxes, segmentation_mask, bbox)
#         except:
#             pass
#             # traceback.print_exc()

#     scores = {
#         "overall": 0.6 * standard_score + 0.2 * iou_score + 0.1 * format_score + 0.1 * length_score,
#         "standard_score": standard_score,
#         "iou_score": iou_score,
#         "format_score": format_score,
#     }
#     return scores


# def accuracy_reward(response: str, ground_truth: str) -> float:
#     answer = extract_boxed_content(response)
#     return 1.0 if grade_response(answer, ground_truth) else 0.0

# def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")

#     scores = []
#     for reward_input in reward_inputs:
#         response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
#         format_score = format_reward(response)
#         accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
#         scores.append(
#             {
#                 "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
#                 "format": format_score,
#                 "accuracy": accuracy_score,
#             }
#         )

#     return scores
