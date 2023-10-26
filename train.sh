export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python train_net.py --dist-url 'tcp://127.0.0.1:50167' \
    --num-gpus 8 \
    --config-file configs/basketball/oneformer_dinat_large_test_a100_size.yaml \
    OUTPUT_DIR outputs/basketball_dinat_a100_b40 WANDB.NAME basketball_dinat_a100_b40
