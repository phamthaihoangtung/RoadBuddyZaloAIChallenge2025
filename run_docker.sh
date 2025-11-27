docker run --gpus '"device=0"' \
    -v /home/tuannd/RoadBuddyZaloAIChallenge2025/fake_private:/data -v /home/tuannd/RoadBuddyZaloAIChallenge2025/fake_private/result:/result \
    -v /home/tuannd/RoadBuddyZaloAIChallenge2025/configs/config_unsloth_infer_docker.yaml:/code/configs/config_unsloth_infer_docker.yaml \
    --net none \
    zac2025:v1 /bin/bash /code/predict.sh

# docker run --gpus '"device=0"' --network host -it --name zac2025 nvidia/cuda:12.2.0-devel-ubuntu22.04 /bin/bash