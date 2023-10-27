# For [MMSports'23] Instance Segmentation Challenge '23 by the ONEDAY Team

## **Step 1: Prepare the Codebase and Challenge Data**
- [ONEFORMER GitHub](https://github.com/SHI-Labs/OneFormer)
- [CHALLENGE Project](https://eval.ai/web/challenges/challenge-page/2070/overview)
- [CHALLENGE GitHub](https://github.com/DeepSportradar/instance-segmentation-challenge)

## **Step 2: Process Data to Coco Panoptic Format Following ONEFORMER**

## **Step 3: Optimization for Performance**
### A: Configuration
Utilize the configuration file at "configs/basketball/oneformer_dinat_large_test_a100_size.yaml," which supports large input size and iterations suitable for basketball dataset.

### B: Memory
Large input size impose restrictions on batch size, hence, we employ act_checkpoint strategy to double the batch size as in "oneformer/modeling/backbone/dinat.py" line 184-185. 

Activation checkpointing is a technique used to reduce GPU memory usage during training. This is done by avoiding the need to store intermediate activation tensors during the forward pass. Instead, the forward pass is recomputed by keeping track of the original input during the backward pass.

### C: Loss
As the "Occlusion Metric" only accounts for instances that are split into several small regions, we have modified the dice_loss in "oneformer/modeling/criterion.py" line 58-62 by adding weights when the ground truth contains more than 5.

### D: Crop
Due to a significant amount of background in the images, we crop 200 pixels in height during the inference stage, considering the relationship between foreground and background in the training and validation data.

### E: V100->A100
The a100 supports a batch size increase to 32 compared to v100's 8.

## Experiment on test
|    |  OM   |  OIR  |  DPR  |  
| ---|-------|-------|-------|
| A  | 0.397 | 0.819 | 0.484 |
| +B | 0.437 | 0.833 | 0.524 |
| +C | 0.476 | 0.800 | 0.596 |
| +D | 0.477 | 0.819 | 0.582 |
| +E | 0.532 | 0.861 | 0.618 |
