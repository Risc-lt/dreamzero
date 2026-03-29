##### Sinple 2 GPU Launch
cd ~/dreamzero
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --model-path ./checkpoints/DreamZero-DROID

cd ~/dreamzero
source .venv/bin/activate
python test_client_AR.py --port 5000


##### Nsys Launch

CUDA_VISIBLE_DEVICES=4,5 nsys profile \
  --trace=nvtx,cuda,cudnn,cublas \
  --output=results/dreamzero_server_profile \
  --force-overwrite true \
  torchrun --nproc_per_node=2 socket_test_optimized_AR.py \
    --port 5000 \
    --model-path ./checkpoints/DreamZero-DROID

python test_client_AR.py --port 5000

pkill -SIGTERM -f torchrun


##### Memory test

python eval_utils/profile_memory.py --model_path ./checkpoints/DreamZero-DROID 2>&1 | tee /tmp/memory_profile.log