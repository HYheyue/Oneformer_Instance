export task=panoptic
export CUDA_VISIBLE_DEVICES=6

rm -rf infer_result/semantic_inference/*
rm -rf infer_result/instance_inference/*
rm -rf infer_result/panoptic_inference/*
mkdir -p infer_result/semantic_inference/
mkdir -p infer_result/instance_inference/
mkdir -p infer_result/panoptic_inference/

python demo/demo_from_json.py --config-file outputs/basketball_dinat_a100_all/config.yaml \
  --input instance-segmentation-challenge/annotations/challenge.json challenge_pan.json  \
  --output outputs \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS outputs/basketball_dinat_a100_all/model_final.pth
