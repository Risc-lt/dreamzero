##### Sinple 2 GPU Launch
cd ~/dreamzero
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --model-path ./checkpoints/DreamZero-DROID

cd ~/dreamzero
source .venv/bin/activate
python test_client_AR.py --port 5000


##### Nsys Launch

CUDA_VISIBLE_DEVICES=4,5 nsys profile \
  --trace=nvtx,cuda,cudnn,cublas \
  --output=dreamzero_server_profile \
  --force-overwrite true \
  torchrun --nproc_per_node=2 socket_test_optimized_AR.py \
    --port 5000 \
    --model-path ./checkpoints/DreamZero-DROID