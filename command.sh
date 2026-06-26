/root/autodl-tmp/DJLaction4/.conda_djiaction4/bin/python tools/corner_pose_resnet.py train \
  --data datasets/dji_action4_rich_views_aug_train_600 \
  --out runs/corner_resnet18_dilated_hm128_rich_views_aug_600_v2_reproduce \
  --backbone resnet18_dilated \
  --image-size 256 \
  --heatmap-size 128 \
  --sigma 2.5 \
  --roi \
  --square-roi \
  --roi-pad 0.08 \
  --augment \
  --epochs 80 \
  --batch-size 16

/root/autodl-tmp/DJLaction4/.conda_djiaction4/bin/python tools/infer_video_yolo_resnet_pose.py \
  --corner-ckpt runs/corner_resnet18_dilated_hm128_rich_views_aug_600_v2_reproduce/best.pt \
  --out runs/corner_resnet18_dilated_hm128_rich_views_aug_600_v2_reproduce/head_left_rgb_raw_yolo_pose.mp4 \
  --backbone resnet18_dilated \
  --image-size 256 \
  --heatmap-size 128 \
  --roi-pad 0.08

/root/autodl-tmp/DJLaction4/.conda_djiaction4/bin/python tools/corner_pose_resnet.py train \
  --data datasets/dji_action4_rich_views_aug_train_600 \
  --out runs/corner_teacher_heatmap_hm128_rich_views_aug_600_geo_reproduce \
  --backbone teacher_heatmap \
  --image-size 256 \
  --heatmap-size 128 \
  --sigma 2.5 \
  --roi \
  --square-roi \
  --roi-pad 0.08 \
  --augment \
  --epochs 120 \
  --batch-size 16 \
  --lr 0.001 \
  --weight-decay 0.0 \
  --scheduler none \
  --heatmap-loss mse_bce \
  --coord-loss-weight 0.01 \
  --edge-loss-weight 0.02 \
  --decode-mode argmax \
  --early-stop-patience 15 \
  --early-stop-min-delta 0.0005

/root/autodl-tmp/DJLaction4/.conda_djiaction4/bin/python tools/infer_video_yolo_resnet_pose.py \
  --corner-ckpt runs/corner_teacher_heatmap_hm128_rich_views_aug_600_geo_reproduce/best.pt \
  --out runs/corner_teacher_heatmap_hm128_rich_views_aug_600_geo_reproduce/head_left_rgb_raw_yolo_pose_argmax.mp4 \
  --backbone teacher_heatmap \
  --image-size 256 \
  --heatmap-size 128 \
  --roi-pad 0.08 \
  --decode-mode argmax