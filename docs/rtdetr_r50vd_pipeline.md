# RT-DETR R50VD Pipeline

Open `rtdetr_r50vd_pipeline.svg` to view the visual pipeline.

This image is redrawn in the same architectural style as the RT-DETR paper overview:

- left: input image and `PResNet-50vd`
- middle: efficient hybrid encoder
- right: uncertainty-minimal query selection and decoder/head

The model path in code is:

```text
Input image
  -> PResNet-50vd backbone
  -> HybridEncoder
  -> RTDETRTransformer decoder
  -> pred_logits + pred_boxes
```

During training, predictions go to:

```text
HungarianMatcher -> SetCriterion -> loss_vfl + loss_bbox + loss_giou
```

During inference, predictions go to:

```text
RTDETRPostProcessor -> top 300 detections -> labels, boxes, scores
```

Important config values:

```text
input size: 640
feature strides: 8, 16, 32
hidden_dim: 256
num_queries: 300
decoder layers: 6
NMS: not used
```
