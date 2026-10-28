#!/usr/bin/env bash
# Single-image-to-scene (i2s) inference tuned for a Colab T4.
#
# Required environment variables:
#   CKPT_DIR        SpatialGen-1.0 checkpoint directory
#   TEST_DATA_DIR   SpatialGen-Testset directory (contains scene_xxxx folders)
# Optional:
#   TEST_SPLIT_FILE split file listing scene ids (default: $TEST_DATA_DIR/test_split.txt)
#   SCENE_ID        scene to generate (default: scene_00000)
#   OUTPUT_DIR      output directory (default: ./out)
#   NUM_STEPS       diffusion steps (default: 20)
set -euo pipefail

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# Makes CUDA errors synchronous; useful while bringing up a new runtime.
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}

: "${CKPT_DIR:?set CKPT_DIR to the SpatialGen-1.0 checkpoint directory}"
: "${TEST_DATA_DIR:?set TEST_DATA_DIR to the SpatialGen-Testset directory}"
TEST_SPLIT_FILE=${TEST_SPLIT_FILE:-$TEST_DATA_DIR/test_split.txt}
SCENE_ID=${SCENE_ID:-scene_00000}
OUTPUT_DIR=${OUTPUT_DIR:-./out}
NUM_STEPS=${NUM_STEPS:-20}

python3 src/inference_sd.py \
    --config_file configs/test_spatialgen_sd21.yaml \
    --tag Img23D_exp_spatialgen \
    --seed 2025 \
    --infer_tag=Img23D_exp_spatialgen_16view_colab \
    --guidance_scale 3.5 \
    --scene_id="$SCENE_ID" \
    --num_inference_steps "$NUM_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --skip_gs \
    opt.pretrained_model_name_or_path="$CKPT_DIR" \
    opt.test_data_dir="$TEST_DATA_DIR" \
    opt.test_split_file="$TEST_SPLIT_FILE" \
    opt.num_inference_steps=$NUM_STEPS \
    opt.input_res=512 \
    opt.num_input_views=1 \
    opt.num_views=16 \
    opt.prediction_type=v_prediction \
    opt.use_layout_prior=true \
    opt.use_scene_coord_map=true \
    opt.use_metric_depth=false \
    opt.input_concat_binary_mask=true \
    opt.input_concat_warpped_image=true \
    opt.spatiallm_data_dir=./none \
    opt.structured3d_data_dir=./none \
    opt.trajectory_sampler_type=spiral
