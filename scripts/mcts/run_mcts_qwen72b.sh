#!/bin/bash

NUM_GPUS=8 # set this to >=4 for 72b models, 1-2 for 3b,7b models
NUM_PROCESSES=10
dataset="web_grounding" # sat2, web_grounding, vstar, web_action
PORT=9001 # port for vllm server

# export port so src/vlmsearch/models/qwen_vllm.py can find the vllm server
export PORT

if [ "$dataset" == "sat2" ]; then
  SYSTEM_PROMPT="""You are a helpful assistant tasked with answering a question about an image. You should systematically reason through the problem step by step by checking and verifying relevant image regions, while grounding reasoning steps to specific (x, y) points in the image:\nEach reasoning step must be enclosed within '<think>' tags and reference exactly one specific coordinate (x, y):\n<think>\n{Single reasoning step with a grounded point} (x, y).\n</think>\nWhen ready to provide the final answer, enclose it within '<answer>' tags:\n<answer> {text of final answer} </answer>\nYour task is to help the user answer the question that may involve small details in the image.\n- Generate ONLY ONE reasoning step OR the final answer per response.\n- Regions are distinct, non-overlapping areas (e.g., quadrants like top-left, small elements or objects, zones like background/foreground).\n- Each step should describe the region then evaluate it for its relevance to the task and to previous steps.\n- Never repeat coordinates from previous steps.\n- Begin by exploring diverse regions, even if they seem less likely, to ensure comprehensive coverage before narrowing down.\n- Prioritize broad coverage of diverse candidates before deciding.\n- Aim for accurate, representative points in the described area/element/object.\n- If unclear, infer based on likely context or purpose.\n- Your final answer should be the text of the choice you think is most correct.\n- Verify each step by examining multiple possible solutions before selecting a final coordinate.\n- Format points as (x, y)"""

  DATA_FILE="$DATA_ROOT/spatial_reasoning/vigorl_sat2_MCTS.jsonl"

  SAVE_TAG="MCTS_SAT_72b"
  JUDGE="string_match"

elif [ "$dataset" == "web_grounding" ]; then
  SYSTEM_PROMPT="""You are a helpful assistant tasked with grounding an element on a web page. You should systematically reason through the problem step by step by checking and verifying relevant webpage regions, while grounding reasoning steps to specific (x, y) points in the image:\nEach reasoning step must be enclosed within '<think>' tags and reference exactly one specific coordinate (x, y):\n<think>\n{Single reasoning step with a grounded point} (x, y).\n</think>\nWhen ready to provide the final answer, enclose it within '<answer>' tags:\n<answer> (xf, yf) </answer>\nYour task is to help the user identify precise (x,y) coordinates of a described area/element/object based on a description.\n- Generate ONLY ONE reasoning step OR the final answer per response.\n- Regions are distinct, non-overlapping areas (e.g., quadrants like top-left, elements like tree/button, zones like background/foreground).\n- Each step should describe the region then evaluate it for its relevance to the task and to previous steps.\n- Never repeat coordinates from previous steps.\n- Begin by exploring diverse regions, even if they seem less likely, to ensure comprehensive coverage before narrowing down.\n- Prioritize broad coverage of diverse candidates before deciding.\n- Aim for accurate, representative points in the described area/element/object.\n- If unclear, infer based on likely context or purpose.\n- Verify each step by examining multiple possible solutions before selecting a final coordinate.\n- Format points as (x, y)"""

  DATA_FILE="$DATA_ROOT/web_grounding/vigorl_osatlas_MCTS.jsonl"

  SAVE_TAG="MCTS_OSATLAS_72b"
  JUDGE="point_in_bbox"

elif [ "$dataset" == "vstar" ]; then
  SYSTEM_PROMPT="""You are a helpful assistant tasked with answering a question about an image. You should systematically reason through the problem step by step by checking and verifying relevant image regions, while grounding reasoning steps to specific (x, y) points in the image:\nEach reasoning step must be enclosed within '<think>' tags and reference exactly one specific coordinate (x, y):\n<think>\n{Single reasoning step with a grounded point} (x, y).\n</think>\nWhen ready to provide the final answer, enclose it within '<answer>' tags:\n<answer> {final answer} </answer>\nYour task is to help the user answer the question that may involve small details in the image.\n- Generate ONLY ONE reasoning step OR the final answer per response.\n- Regions are distinct, non-overlapping areas (e.g., quadrants like top-left, small elements or objects, zones like background/foreground).\n- Each step should describe the region then evaluate it for its relevance to the task and to previous steps.\n- Never repeat coordinates from previous steps.\n- Begin by exploring diverse regions, even if they seem less likely, to ensure comprehensive coverage before narrowing down.\n- Prioritize broad coverage of diverse candidates before deciding.\n- Aim for accurate, representative points in the described area/element/object.\n- If unclear, infer based on likely context or purpose.\n- Your final answer should be the choice you think is most correct.\n- Verify each step by examining multiple possible solutions before selecting a final coordinate.\n- Format points as (x, y)"""

  DATA_FILE="$DATA_ROOT/visual_search/vigorl_search_MCTS.jsonl"

  SAVE_TAG="MCTS_VSTAR_72b"
  JUDGE="string_match"

elif [ "$dataset" == "web_action" ]; then

  SYSTEM_PROMPT="""You are a helpful Assistant tasked with navigating a web browser. These tasks will be accomplished through the use of specific actions you can issue. Your task is to choose the action that makes the most progress towards an objective. You should systematically reason through the problem step by step by checking and verifying possible actions and webpage regions, while grounding reasoning steps to specific (x, y) points in the image:\nEach reasoning step must be enclosed within '<think>' tags and reference exactly one specific coordinate (x, y):\n<think>\n{Single reasoning step with a grounded point} (x, y).\n</think>\nWhen ready to provide the final answer, enclose it within '<answer>' tags:\n<answer> {action} </answer>\n- Generate ONLY ONE reasoning step OR the final action per response.\n- Each reasoning step must explicitly describe and evaluate the region’s relevance to the objective and proposing an action.\n- Never repeat coordinates from previous steps.\n- Look at diverse webpage regions to figure out which action should be taken.\n- Verify your selection by examining multiple possible solutions.

**Inputs**
Here's the information you'll have:
1. OBJECTIVE: This is the task you are trying to complete.
2. The web page screenshot: This is a screenshot of the current webpage you are on, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.
3. PREVIOUS ACTIONS: This is the actions that you have performed prior to getting to the current page, but instead of the button id, the button text of the actions taken on the previously navigated pages are provided.

**Action Space**
You can take the following actions:
1. ```click [id]```: This action clicks on an element with a specific id on the webpage.
2. ```type [id] [content]```: Use this to type the content into the field with id. By default, typing the content simulates pressing the "Enter" key afterward to submit the text.
3. ```scroll [down]```: Scroll the page up or down.
4. ```go_back```: Navigate to the previously viewed page.
5. ```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If no answer is required, output empty brackets.

**Guidelines**
To be successful, it is very important to follow the following rules:
2. Generate the final action in the correct format. For example, '<answer> click [1234] </answer>'.
3. Issue the stop action (i.e. stop [answer]) when you think you have achieved the objective. Don't generate anything after stop.
4. In your final answer, you should only output a single action and should never output a prediction involving taking multiple actions.
5. Reference of image regions should be formatted as '(x, y)', where x and y are the center image coordinates of the region.
6. You should output atleast 3 reasoning steps before issuing the final action.
"""
  
  DATA_FILE="$DATA_ROOT/web_action/vigorl_web_action_MCTS.jsonl"

  SAVE_TAG="MCTS_WEB_ACTION_72b"
  JUDGE="web_action"
fi

IMAGE_ROOT=$DATA_ROOT
MODEL="Qwen/Qwen2.5-VL-72B-Instruct"

ACTOR_MODEL="qwen_vllm"

# Add a cleanup function to kill the vLLM server on script exit
cleanup() {
  echo "[INFO] Cleaning up resources..."
  # Find and kill the vLLM server process
  if pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "[INFO] Killing vLLM server process..."
    pkill -f "vllm serve"
  fi
  exit 0
}

# # Set trap to call cleanup function on script exit, including Ctrl+C (SIGINT)
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------
# 1) If ACTOR_MODEL is "qwen_vllm", check if the vllm server is running.
#    If not running, start serve_qwen.sh in the background.
#    Then wait for it to become reachable.
# ---------------------------------------------------------------------
if [ "$ACTOR_MODEL" == "qwen_vllm" ]; then
  # Simple check: see if something is already listening on port 8000
  # or if "vllm serve" is in the process list
  if lsof -Pi :9001 -sTCP:LISTEN -t >/dev/null 2>&1 || pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "[INFO] vllm server is already running on port 9001"
  else
    echo "[INFO] Starting vllm server via serve_qwen.sh ..."
    echo "[INFO] VLLM logs saving to vllm_logs/ ..."
    # You can pass $MODEL and GPU count from environment or SLURM variables
    bash scripts/vllm/serve_qwen.sh "$MODEL" "${NUM_GPUS}" "${PORT}" &
    sleep 5  # brief pause before checking
  fi

  # ----------------------------------------------------------------
  # 2) Wait for the vllm server to be reachable on port 8000
  #    We'll try up to 30 times, sleeping 5 seconds each time.
  # ----------------------------------------------------------------
  echo "[INFO] Please wait for vllm server to be responsive..."
  for i in {1..360}; do
    if curl --max-time 2 -s -o /dev/null http://localhost:9001/v1/models; then
      echo "[INFO] vllm server is up!"
      break
    else
      echo "[WARN] checked vllm server but it is not up yet... retry ($i/360) in 10s"
      sleep 10
    fi

    # if last attempt also fails, exit
    if [ "$i" -eq 360 ]; then
      echo "[ERROR] vllm server did not respond after many attempts, exiting."
      exit 1
    fi
  done
fi

python -m src.vlmsearch \
    --num_processes=${NUM_PROCESSES} \
    --use_python_mp \
    --model ${ACTOR_MODEL} \
    --seed 42 \
    --judge ${JUDGE} \
    --search_method mcts \
    --max_depth 10 \
    --n_simulations 8 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_new_tokens 512 \
    --checkpoint_interval 1 \
    --save_rollouts \
    --n_rollouts_per_node 2 \
    --num_children_per_expand 3 \
    --c_puct 2.0 \
    --save_rollouts_dir "data/mcts" \
    --add_thought_number_system_prompt \
    --system_prompt "${SYSTEM_PROMPT}" \
    --pretrained "${MODEL}" \
    --image_root "${IMAGE_ROOT}" \
    --data_files "${DATA_FILE}" \
    --save_tag "${SAVE_TAG}"
