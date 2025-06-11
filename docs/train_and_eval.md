### 4.3 Training

Training DynRefer on VG V1.2.
```
bash scripts/train/train_vg1.2.sh
```
Finetuning DynRefer on RefCOCOg (Pretrained by VG V1.2).
```
bash scripts/train/finetune_refcocog.sh ckpts/vg1.2_5e.pth
```

Training the GRPO view selector (requires a pretrained DynRefer checkpoint).
```
bash scripts/train/train_grpo.sh configs/train/vg/vg1.2_5e.yaml
```



### 4.4 Evaluation
Evaluating the dense captioning performance of DynRefer on VG V1.2 : (`mAP 47.6`).
```
bash scripts/eval/eval_vg1.2_densecap.sh ckpts/vg1.2_5e.pth
```
Evaluating the region-level captioning performance of DynRefer on VG : (`METEOR 21.2, CIDEr 190.5`).
```
bash scripts/eval/eval_vg_reg.sh ckpts/vg1.2_5e.pth
```
Evaluating the region-level captioning performance of DynRefer on RefCOCOg : (`METEOR 18.1, CIDEr 115.3`).
```
bash scripts/eval/eval_refcocog_reg.sh ckpts/refcocog_ft.pth
```
Evaluating the attribute detection performance of DynRefer on OVAD : (`mAP 29.2`).
```
bash scripts/eval/eval_ovad.sh ckpts/vg1.2_5e.pth
```
Evaluating the region recognition performance of DynRefer on COCO : (`Acc. 91.8`).
```
bash scripts/eval/eval_coco.sh ckpts/vg1.2_5e.pth
```


