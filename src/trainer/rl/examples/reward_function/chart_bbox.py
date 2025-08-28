from ast import Tuple
import json
import re
from typing import List, Tuple

from mathruler.grader import extract_boxed_content, grade_answer
from typing import Dict
from examples.reward_function.chart_multibbox import calculate_bbox_iou


# def format_reward(response: str) -> float:
#     pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
#     format_match = re.fullmatch(pattern, response)
#     return 1.0 if format_match else 0.0


def format_reward(predict_str: str) -> float:
    # Check for proper format with think and answer tags
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    format_match = re.fullmatch(pattern, predict_str, re.DOTALL)
    
    if not format_match:
        return 0.0
    
    # Extract the answer content
    content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if not content_match:
        return 0.0
    
    return 1.0

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
            # breakpoint()
            gt_float = float(ground_truth_formatted)
            pred_float = float(predicted_numbers[0])
            return abs(gt_float - pred_float) <= error_margin
            
    except (ValueError, TypeError, IndexError):
        # If number parsing fails, fall back to string comparison
        return ground_truth == predicted_answer


def accuracy_reward_with_error(predict_str: str, ground_truth: str, ground_truth_precision, error_margin) -> float:
    # answer = extract_boxed_content(response)
    content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
    pred_ans = content_match.group(1).strip() if content_match else predict_str.strip()
    return 1.0 if grade_response(pred_ans, ground_truth, ground_truth_precision, error_margin) else 0.0


def coordinate_reward(predict_str: str) -> float:
    """
    Checks if the thinking block contains at least 2 different coordinate pairs in format (x, y).
    
    INPUTS:
    - predict_str: The full prediction string including think and answer tags.
    
    OUTPUTS:
    - score: 1.0 if at least 2 different coordinate pairs are found in the thinking block, 0.0 otherwise.
    """
    # Extract the thinking content
    think_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    if not think_match:
        return 0.0
    
    thinking_content = think_match.group(1)
    # Search thinking content for bounding boxes
    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_pattern, thinking_content)

    if not matches:
        return 0.0
    coords = []
    # Try to parse each match as JSON
    for match in matches:
        # breakpoint()
        try:
            parsed_json = json.loads(match.strip())
            for item in parsed_json:
                if "bbox_2d" in item:
                    coords.append(item["bbox_2d"])
        except json.JSONDecodeError:
            continue
    return len(coords) >= 1
    
    # Find all coordinate pairs in the format (x, y)
    coordinates = re.findall(r"\((\d+)\s*,\s*(\d+)\)", thinking_content)
    
    # Convert to set of tuples to get unique coordinates
    unique_coordinates = set(tuple(map(int, coord)) for coord in coordinates)
    
    # Return 1.0 if there are at least 1 different coordinate pairs
    return int(len(unique_coordinates) >= 1)


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

# def extract_bbox_from_response(text):
#     """
#     Extract JSON content from markdown code blocks in the response.

#     Args:
#         text: The model's response text

#     Returns:
#         Parsed JSON object or None if no valid JSON found
#     """
#     # Find content between ```json and ```
#     json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
#     matches = re.findall(json_pattern, text)

#     if not matches:
#         return None
#     # breakpoint()
#     coords = []
#     # Try to parse each match as JSON
#     for match in matches:
#         try:
#             parsed_json = json.loads(match.strip())
#             for item in parsed_json:
#                 if "bbox_2d" in item:
#                     coords.append(item["bbox_2d"])
#         except json.JSONDecodeError:
#             continue
#     return coords
#     # If we couldn't parse any match as valid JSON, try with ast.literal_eval
#     import ast
#     for match in matches:
#         try:
#             # Clean up the match a bit
#             cleaned = match.strip().replace("'", "\"")
#             parsed_json = ast.literal_eval(cleaned)
#             return parsed_json
#         except:
#             continue

#     return None

def extract_bbox_from_response(text: str) -> List[List[int]]:
    """
    Extract bounding boxes from LLM responses as tuples or lists of 4 integers.
    
    Supports:
    - Tuples like (x1, y1, x2, y2)
    - Lists like [x1, y1, x2, y2]

    Args:
        text: The model's response text

    Returns:
        List of bounding box coordinates as (x1, y1, x2, y2) tuples
    """
    coords = []
    
    # Pattern for tuples: (x1, y1, x2, y2)
    tuple_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
    tuple_matches = re.findall(tuple_pattern, text)
    
    for match in tuple_matches:
        coords.append(list(map(int, match)))
    
    # Pattern for lists: [x1, y1, x2, y2]
    list_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    list_matches = re.findall(list_pattern, text)
    
    for match in list_matches:
        coords.append(list(map(int, match)))
    
    return coords



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
    iou_score = 0
    pred_bboxes = extract_bbox_from_response(predict_str)
    coordinate_score = len(pred_bboxes) >= 1 
    if pred_bboxes:
        iou_score = calculate_bbox_iou(pred_bboxes, gt_bbox=extra_info["bbox"])
    return {
        "overall": 0.6 * accuracy_score + 0.3 * iou_score + 0.1 * (format_score * coordinate_score),
        "format": format_score,
        "coordinate": coordinate_score,
        "iou": iou_score,
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

# Test cases
def test_extract_bbox_from_response():
    
    # Basic tuple tests
    test_cases = [
        # Single tuple - basic
        ("The object is at (10, 20, 100, 200)", [[10, 20, 100, 200]]),
        
        # Single list - basic  
        ("The bounding box is [50, 75, 150, 175]", [[50, 75, 150, 175]]),
        
        # Multiple tuples
        ("Found objects at (10, 20, 30, 40) and (50, 60, 70, 80)", 
         [[10, 20, 30, 40], [50, 60, 70, 80]]),
        
        # Multiple lists
        ("Boxes: [10, 20, 30, 40], [100, 200, 300, 400]", 
         [[10, 20, 30, 40], [100, 200, 300, 400]]),
        
        # Mixed tuples and lists
        ("Detection results: (10, 20, 30, 40) and [50, 60, 70, 80]",
         [[10, 20, 30, 40], [50, 60, 70, 80]]),
        
        # With varying whitespace
        ("Coordinates: ( 10 , 20 , 30 , 40 ) and [ 50 , 60 , 70 , 80 ]",
         [[10, 20, 30, 40], [50, 60, 70, 80]]),
        
        # No whitespace
        ("Tight format: (10,20,30,40) and [50,60,70,80]",
         [[10, 20, 30, 40], [50, 60, 70, 80]]),
        
        # Large numbers
        ("Big coordinates: (1000, 2000, 3000, 4000)",
         [[1000, 2000, 3000, 4000]]),
        
        # Zero coordinates
        ("Starting at origin: (0, 0, 100, 100)",
         [[0, 0, 100, 100]]),
        
        # Empty string
        ("", []),
        
        # No coordinates
        ("This text has no bounding boxes in it", []),
        
        # Partial matches (should not match)
        ("Incomplete: (10, 20, 30) or [40, 50]", []),
        
        # Too many numbers (should not match)
        ("Too many: (10, 20, 30, 40, 50) or [60, 70, 80, 90, 100]", []),
        
        # Nested in sentences
        ("The person detected at (100, 150, 200, 250) is walking towards the car at [300, 400, 500, 600].",
         [[100, 150, 200, 250], [300, 400, 500, 600]]),
        
        # With negative signs (should not match since pattern looks for \d+)
        ("Negative coords: (-10, -20, 30, 40)", []),
        
        # With decimal points (should not match)
        ("Decimal coords: (10.5, 20.5, 30.5, 40.5)", []),
        
        # Multiple on same line
        ("Objects: (10, 20, 30, 40), (50, 60, 70, 80), [90, 100, 110, 120]",
         [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]),
        
        # Mixed with other parentheses/brackets
        ("Function call foo(a, b) and array[i] with bbox (10, 20, 30, 40)",
         [[10, 20, 30, 40]]),
        
        # Realistic LLM response
        ("""I found 2 objects in the image:
         1. A person at coordinates (145, 67, 289, 456)  
         2. A car located at [523, 234, 789, 445]
         The confidence scores are 0.95 and 0.87 respectively.""",
         [[145, 67, 289, 456], [523, 234, 789, 445]]),

          # Realistic LLM response
        ("""I found 2 objects in the image:
         1. A person at coordinates 
         ```json
        [(145, 67, 289, 456)  
         ```
         2. A car located at [523, 234, 789, 445]
         The confidence scores are 0.95 and 0.87 respectively.""",
         [[145, 67, 289, 456], [523, 234, 789, 445]])
    ]
    
    print("Running tests...")
    for i, (input_text, expected) in enumerate(test_cases):
        result = extract_bbox_from_response(input_text)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"Test {i+1}: {status}")
        if result != expected:
            print(f"  Input: {repr(input_text)}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
        print()

if __name__ == "__main__":
    test_extract_bbox_from_response()