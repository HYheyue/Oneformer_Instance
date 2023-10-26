export task=panoptic
export CUDA_VISIBLE_DEVICES=0

rm -rf infer_result/semantic_inference/*
rm -rf infer_result/instance_inference/*
rm -rf infer_result/panoptic_inference/*

OUTPUTS="outputs/basketball_dinat_a100_all/"
MODEL="model_final.pth"
for model in $MODEL
do
echo ${model}
OUTPUT=${model%.*}
python demo/demo_from_json.py \
  --config-file ${OUTPUTS}config.yaml \
  --input instance-segmentation-challenge/annotations/test.json ${OUTPUT}_test_pan.json \
  --output outputs --task $task --opts MODEL.IS_TRAIN False \
  MODEL.IS_DEMO True MODEL.WEIGHTS ${OUTPUTS}${model}
done
