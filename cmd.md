##### Sinple 2 GPU Launch
cd ~/dreamzero
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --standalone --nproc_per_node=2 \
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


##### ncu Kernel Profiling (Roofline — compute-bound vs memory-bound)
# Terminal 1: launch server under ncu, targeting only the dit_forward NVTX range
mkdir -p results
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2,3 ncu \
  --nvtx \
  --nvtx-include "dit_forward]" \
  --set roofline \
  --target-processes all \
  -o results/dreamzero_ncu \
  -f \
  torchrun --nproc_per_node=2 socket_test_optimized_AR.py \
    --port 5000 --enable-dit-cache \
    --model-path ./checkpoints/DreamZero-DROID

# Terminal 2: send a request to trigger inference (do this twice — first is KV prefill, second is cached)
python test_client_AR.py --port 5000

# View report in GUI (transfer .ncu-rep to a desktop machine):
ncu-ui results/dreamzero_ncu.ncu-rep

# Or export headless CSV summary:
ncu --import results/dreamzero_ncu.ncu-rep --csv --page details > results/ncu_summary.csv


##### Memory test

python eval_utils/profile_memory.py --model_path ./checkpoints/DreamZero-DROID 2>&1 | tee /tmp/memory_profile.log