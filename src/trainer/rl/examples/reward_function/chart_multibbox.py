import math
import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer
from typing import Dict
import json


# def format_reward(response: str) -> float:
#     pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
#     format_match = re.fullmatch(pattern, response)
#     return 1.0 if format_match else 0.0



_TAG_RE = re.compile(r"<(/?)(tool_call|observation|think|answer)>", re.IGNORECASE)
# _TOOL_JSON_RE = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", re.DOTALL)
_TOOL_JSON_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

def format_reward(predict_str: str):
    s = predict_str.strip()

    # 1) must finish with </answer> (optionally followed by one EOS token)
    if not re.search(r"</answer>\s*(<\|im_end\|>)?\s*$", s, re.DOTALL):
        return 0.0

    # 2) walk through the high‑level tag sequence to enforce grammar
    tags_iter = _TAG_RE.finditer(s)
    state = "think_open"            # expected next tag
    for m in tags_iter:
        tag = m.group(0).lower()

        if state == "tool_open":
            if tag != "<tool_call>":
                return 0.0
            state = "tool_close"

        elif state == "tool_close":
            if tag != "</tool_call>":
                return 0.0
            state = "obs_open"

        elif state == "obs_open":
            if tag != "<observation>":
                return 0.0
            state = "obs_close"

        elif state == "obs_close":
            if tag != "</observation>":
                return 0.0
            state = "think_open"

        elif state == "think_open":
            if tag != "<think>":
                return 0.0
            state = "think_close"

        elif state == "think_close":
            if tag != "</think>":
                return 0.0
            state = "post_think"

        elif state == "post_think":
            if tag == "<tool_call>":
                state = "tool_close"         # start another round
            elif tag == "<answer>":
                state = "answer_close"
            else:
                return 0.0

        elif state == "answer_close":
            if tag != "</answer>":
                return 0.0
            state = "end"

        elif state == "end":
            # no structural tags allowed after </answer>
            return 0.0

    if state != "end":
        return 0.0   # we never saw a complete <answer> … </answer> block

    # 3) validate each <tool_call> JSON and coordinate schema
    # Also track unique coordinates for reward calculation
    previous_coords = []
    min_distance_threshold = 10  # Minimum distance in pixels between coordinates
    
    for m in _TOOL_JSON_RE.finditer(s):
        try:
            obj = json.loads(m.group(1))
            arguments = obj.get("arguments", {})
            coord = arguments.get("bbox_2d", None)
            label = arguments.get("label", None)
            if (not isinstance(coord, list) or len(coord) != 4 or
                not all(isinstance(x, int) for x in coord)):
                return 0.0
            if label is None:
                return 0.0
            
            # Add valid coordinate to our tracking list
            previous_coords.append(coord)
        except Exception:
            return 0.0

    # 4) validate final answer is a tuple of two ints
    ans_match = re.search(r"<answer>\s*\(([^)]*)\)\s*</answer>", s)
    if not ans_match:
        return 0.0
    # try:
    #     ans_tuple = ast.literal_eval("(" + ans_match.group(1).strip() + ")")
    #     if (not isinstance(ans_tuple, tuple) or len(ans_tuple) != 2 or
    #         not all(isinstance(x, int) for x in ans_tuple)):
    #         return 0.0
    # except Exception:
    #     return 0.0
    return 1.0
    # 5) base reward + bonus for extra turns with sufficient diversity
    reward = 1.0

    # Count unique and sufficiently distant coordinates
    unique_coords = []
    for coord in previous_coords:
        # Check if this coordinate is too close to any we've already counted
        too_close = False
        for existing_coord in unique_coords:
            # Calculate Euclidean distance
            distance = math.sqrt((coord[0] - existing_coord[0])**2 + 
                                 (coord[1] - existing_coord[1])**2)
            if distance < min_distance_threshold:
                too_close = True
                break
        
        if not too_close:
            unique_coords.append(coord)
    
    # Award bonus only for unique, sufficiently distant coordinates
    # TODO: should we keep this?
    num_unique_turns = len(unique_coords)
    if num_unique_turns > 1:
        reward += 0.2 * (num_unique_turns - 1)

    return reward

def extract_bbox_from_response(text):
    s = text.strip()
    coords = []
    for m in _TOOL_JSON_RE.finditer(s):
        try:
            obj = json.loads(m.group(1))
            arguments = obj.get("arguments", {})
            coord = arguments.get("bbox_2d", None)
            label = arguments.get("label", None)
            if (not isinstance(coord, list) or len(coord) != 4 or
                not all(isinstance(x, int) for x in coord)):
                return 0.0
            if label is None:
                return 0.0
            
            # Add valid coordinate to our tracking list
            coords.append(coord)
        except Exception:
            continue
    return coords


def extract_numbers_from_string(text):
    """
    Extract all numbers from a string.
    
    Args:
        text: Input string that may contain numbers
    
    Returns:
        list: List of number strings found in the text (empty list if none found)
    """
    # Pattern to match numbers (including negative, decimals, scientific notation)
    number_pattern = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
    
    # Find all numbers in the string
    numbers = re.findall(number_pattern, str(text).strip())
    
    return numbers


def format_numbers_with_precision(numbers, precision):
    """
    Format number(s) according to the specified precision.
    
    Args:
        numbers: Single number string, list of number strings, or non-numeric string
        precision: Integer precision level or None for categorical
    
    Returns:
        Formatted result - single string, list of strings, or original string
    """
    def format_single_number(num_str):
        """Format a single number string according to precision."""
        rounded_number = round(float(num_str), precision)
        if precision <= 0:
            return str(int(rounded_number))
        return f"{rounded_number:.{precision}f}"
    
    # Handle categorical/text answers
    if precision is None:
        return str(numbers).strip() if not isinstance(numbers, list) else numbers
    
    # Handle single number string
    if isinstance(numbers, str) and numbers:
        try:
            return format_single_number(numbers).strip()
        except (ValueError, TypeError):
            return numbers
    
    # Handle list of numbers
    if isinstance(numbers, list):
        if len(numbers) == 0:
            return str(numbers) if not isinstance(numbers, list) else numbers
        elif len(numbers) == 1:
            return format_single_number(numbers[0]).strip()
        else:
            return [format_single_number(num).strip() for num in numbers]
    
    # Fallback for other types
    return str(numbers).strip()


def get_answer_from_precision(answer, ground_truth_precision):
    """
    Get the ground truth answer formatted according to precision.
    
    Args:
        answer: The ground truth answer (string or number)
        ground_truth_precision: Integer precision level or None for categorical
    
    Returns:
        Formatted ground truth answer
    """
    # Handle categorical/text answers
    if ground_truth_precision is None:
        return str(answer).strip()
    
    # Extract numbers from the answer
    numbers = extract_numbers_from_string(answer)
    
    if not numbers:
        # No numbers found, return original string
        return str(answer).strip()
    
    # Format the numbers with the specified precision
    return format_numbers_with_precision(numbers, ground_truth_precision)



def grade_response(ground_truth, predicted_answer, ground_truth_precision, error_margin):
    """
    Grade the response based on ground truth and predicted answer with error margin tolerance.
    
    Args:
        ground_truth: The ground truth answer (string)
        predicted_answer: The predicted answer (string) 
        ground_truth_precision: Integer precision level or None for categorical
        error_margin: Acceptable error margin for numerical answers
    
    Returns:
        bool: True if the prediction is correct within the error margin
    """
    # Convert inputs to strings
    ground_truth = str(ground_truth).strip()
    predicted_answer = str(predicted_answer).strip()
    
    if ground_truth_precision is None:
        # Exact string matching for categorical answers
        return ground_truth == predicted_answer
    
    try:
        # Format ground truth with the specified precision
        ground_truth_formatted = get_answer_from_precision(ground_truth, ground_truth_precision)
        
        # Extract numbers from predicted answer (don't format with precision)
        predicted_numbers = extract_numbers_from_string(predicted_answer)
        
        # Handle cases where ground truth is formatted as a list vs single value
        if isinstance(ground_truth_formatted, list):
            # Ground truth has multiple numbers
            if len(predicted_numbers) != len(ground_truth_formatted):
                return False
            
            # Compare each number pair within error margin
            for gt_str, pred_str in zip(ground_truth_formatted, predicted_numbers):
                gt_float = float(gt_str)
                pred_float = float(pred_str)
                if abs(gt_float - pred_float) > error_margin:
                    return False
            return True
            
        else:
            # Ground truth is a single number
            if len(predicted_numbers) != 1:
                return False
            
            gt_float = float(ground_truth_formatted)
            pred_float = float(predicted_numbers[0])
            return abs(gt_float - pred_float) <= error_margin
            
    except (ValueError, TypeError, IndexError):
        # If number parsing fails, fall back to string comparison
        return ground_truth == predicted_answer

# def accuracy_reward(response: str, ground_truth: str) -> float:
#     answer = extract_boxed_content(response)
#     return 1.0 if grade_answer(answer, ground_truth) else 0.0

def accuracy_reward_with_error(predict_str: str, ground_truth: str, ground_truth_precision, error_margin) -> float:
    # answer = extract_boxed_content(response)
    content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
    pred_ans = content_match.group(1).strip() if content_match else predict_str.strip()
    return 1.0 if grade_response(pred_ans, ground_truth, ground_truth_precision, error_margin) else 0.0


def calculate_bbox_iou(pred_bboxes,  gt_bbox=None):
    """
    Calculate IoU between predicted bounding boxes and ground truth (segmentation mask or bbox).

    Args:
        pred_bboxes: List of predicted bounding boxes in format [x1, y1, x2, y2]
        seg_mask: Ground truth segmentation mask tensor
        gt_bbox: Ground truth bounding box in format [x1, y1, x2, y2]

    Returns:
        Mean IoU score across all bounding boxes
    """
    if not pred_bboxes:
        return 0.0

    # If single layer bbox, wrap it in a list
    if not isinstance(pred_bboxes[0], list):
        pred_bboxes = [pred_bboxes]

    # if seg_mask is not None and isinstance(seg_mask, numpy.ndarray):
    #     seg_mask = torch.from_numpy(seg_mask)

    # # Not none and not all zero
    # if seg_mask is not None and torch.sum(seg_mask) > 0:
    #     # Get mask dimensions
    #     if len(seg_mask.shape) == 3:  # Channel dimension
    #         height, width = seg_mask.shape[1], seg_mask.shape[2]
    #     else:
    #         height, width = seg_mask.shape[0], seg_mask.shape[1]

    #     # Convert segmentation mask to binary (1 for any positive value)
    #     binary_seg_mask = (seg_mask > 0).float()

    #     total_iou = 0.0
    #     for bbox in pred_bboxes:
    #         if len(bbox) < 4:
    #             continue
    #         # Convert bbox to mask
    #         try:
    #             bbox_mask = bbox_to_mask(bbox, height, width)
    #         except:
    #             continue

    #         # Calculate intersection and union
    #         intersection = torch.sum(bbox_mask * binary_seg_mask)
    #         union = torch.sum(torch.clamp(bbox_mask + binary_seg_mask, 0, 1))

    #         # Calculate IoU
    #         iou = intersection / union if union > 0 else 0.0
    #         total_iou += iou

    #     # Return mean IoU
    #     return total_iou / len(pred_bboxes)
    breakpoint()
    if gt_bbox is not None:
        # Calculate IoU directly between bounding boxes
        # total_iou = 0.0
        max_iou = 0.0
        for pred_bbox in pred_bboxes:
            if len(pred_bbox) < 4:
                continue
            # Calculate intersection
            # gt_bbox = gt_bbox.tolist()
            # print("pred_bbox: ", pred_bbox.__class__)
            # print("gt_bbox: ", gt_bbox.__class__)
            x1 = max(pred_bbox[0], gt_bbox[0])
            y1 = max(pred_bbox[1], gt_bbox[1])
            x2 = min(pred_bbox[2], gt_bbox[2])
            y2 = min(pred_bbox[3], gt_bbox[3])

            # Check if boxes overlap
            if x1 >= x2 or y1 >= y2:
                iou = 0.0
            else:
                # Calculate areas
                intersection = (x2 - x1) * (y2 - y1)
                pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
                gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
                union = pred_area + gt_area - intersection

                # Calculate IoU
                iou = intersection / union if union > 0 else 0.0

            max_iou = max(max_iou, iou)

        # Return mean IoU
        return max_iou / len(pred_bboxes)

    else:
        # No ground truth bbox provided
        return 0.0

def compute_score_with_error(predict_str: str, ground_truth: str, extra_info, format_weight: float = 0.1) -> list[dict[str, float]]:
    # if not isinstance(reward_inputs, list):
    #     raise ValueError("Please use `reward_type=batch` for math reward function.")
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)  # handle qwen2.5vl-32b format
    format_score  = format_reward(predict_str)
    accuracy_score = accuracy_reward_with_error(predict_str, ground_truth, 
    extra_info["ground_truth_precision"], extra_info["error_margin"])
    iou_score = 0
    pred_bboxes = extract_bbox_from_response(predict_str)
    if pred_bboxes:
        assert "bbox" in extra_info, "Ground truth bbox is missing in extra_info"
        iou_score = calculate_bbox_iou(pred_bboxes, gt_bbox=extra_info["bbox"])
    return {
        "overall": 0.6 * accuracy_score + 0.3 * iou_score + 0.1 * format_score,
        "format": format_score,
        "iou": iou_score,
        "accuracy": accuracy_score,
    }

# def evaluate_bbox_format(predict_str):
#     """
#     Evaluate the format correctness of the bounding box JSON in the response.
#     Returns a score based on how well the response follows the expected format.

#     Args:
#         predict_str: The model's prediction string

#     Returns:
#         Format score between 0.0 and 1.0
#     """
#     format_score = 0.0

#     # Check if response contains a code block
#     if "```" in predict_str:
#         format_score += 0.2  # 20% for having a code block

#         # Check if it's specifically marked as JSON
#         if "```json" in predict_str:
#             format_score += 0.1  # Additional 10% for correct JSON marker

#     # Try to extract and parse JSON
#     json_str = parse_json(predict_str)
#     if not json_str:
#         return format_score  # Failed to find JSON content

#     try:
#         # Try to parse as JSON
#         parsed_json = None
#         try:
#             parsed_json = json.loads(json_str)
#             format_score += 0.2  # Additional 20% for valid JSON
#         except json.JSONDecodeError:
#             # Try with ast.literal_eval as fallback
#             import ast
#             try:
#                 cleaned = json_str.replace("'", "\"")
#                 parsed_json = ast.literal_eval(cleaned)
#                 format_score += 0.1  # Only 10% for requiring fallback parsing
#             except:
#                 return format_score  # Failed to parse

#         # Check if it's a list of objects
#         if not isinstance(parsed_json, list):
#             return format_score

#         format_score += 0.1  # Additional 10% for being a list

#         # Check each item for proper bbox structure
#         valid_items = 0
#         total_items = len(parsed_json)

#         for item in parsed_json:
#             if not isinstance(item, dict):
#                 continue

#             # Check for required fields
#             has_bbox = "bbox_2d" in item
#             has_label = "label" in item

#             if has_bbox and has_label:
#                 bbox = item["bbox_2d"]
#                 # Check bbox format [x1, y1, x2, y2]
#                 if (isinstance(bbox, list) and len(bbox) == 4 and
#                         all(isinstance(coord, (int, float)) for coord in bbox)):
#                     valid_items += 1

#         # Add up to 40% based on proportion of valid items
#         if total_items > 0:
#             format_score += 0.4 * (valid_items / total_items)

#     except Exception:
#         # Any other parsing issues
#         pass

#     return format_score

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
