# DJI Action 4 项目当前进展与后续规划

## 一、任务目标

根据老师给出的流程，完成一个针对 DJI Action 4 的实例级角点检测原型：

1. 根据目标图片得到目标物体三维模型
2. 使用 Blender / HCCEPose 思路渲染训练数据
3. 实现输入 RGB、输出 8 个角点 heatmap 的网络
4. 在渲染数据上训练
5. 在目标视频上测试

当前范围已经明确为：

- 只做 **2D 八个角点检测**
- **不做 PnP**
- **不做 6D 位姿恢复**

## 二、当前已经完成的内容

### 1. 三维模型准备

已经得到 DJI Action 4 的三维模型，并整理为可用于渲染的数据格式。

关键文件：

- `/root/autodl-fs/dji.glb`
- `datasets/dji_action4/models/obj_000001.ply`
- `datasets/dji_action4/models/models_info.json`
- `datasets/dji_action4/camera.json`

### 2. 合成训练数据渲染

已经实现渲染脚本，并生成了多版合成角点数据。

关键脚本：

- `tools/render_corner_dataset.py`

关键数据：

- `datasets/dji_action4_corner_train_100`
- `datasets/dji_action4_corner_train_500`
- `datasets/dji_action4_corner_train_light_1500_front_yplus`

说明：

- 目前已经具备“从三维模型渲染 RGB + 自动生成 8 角点标签”的能力
- 这一步是整个老师流程中的核心基础环节

### 3. 目标检测阶段

为了在真实视频中先定位相机 ROI，已经补充训练了一个 YOLO 检测器。

关键模型：

- `runs/detect/runs/dji_action4_yolo_real_full_ft/weights/best.pt`

作用：

- 先在整帧视频里找到 DJI Action 4
- 再把 ROI 送入角点网络

说明：

- 这一步不是老师流程里的最终目标
- 它是为了让真实视频上的角点检测更稳定

### 4. 8 角点 heatmap 网络

已经实现并训练了一个输入 ROI RGB、输出 8 个角点 heatmap 的网络。

关键脚本：

- `tools/corner_pose_resnet.py`
- `tools/infer_video_yolo_resnet_pose.py`

当前采用方案：

- backbone：`ResNet18`
- 输出：`8` 通道 heatmap
- 后处理：从 heatmap 中提取 8 个点并连线显示

### 5. 角点编号体系

当前已经确定最终编号规则：

- 镜头面：`0 1 2 3`
- 背面：`4 5 6 7`

具体定义：

- `0`: 镜头面左上
- `1`: 镜头面右上
- `2`: 镜头面右下
- `3`: 镜头面左下
- `4`: 背面左上
- `5`: 背面右上
- `6`: 背面右下
- `7`: 背面左下

参考图：

- `runs/vis_frontback_order_yplus/frontback_order_guide.png`

### 6. 真实数据人工标注与微调

为了缩小“合成数据 -> 真实视频”的差距，已经对真实 ROI 做了多批人工角点标注，并用于微调。

当前已整理的真实标注数据：

- `datasets/dji_action4_real_corner_yplus_labeled`
- `datasets/dji_action4_real_corner_yplus_labeled_batch2`
- `datasets/dji_action4_real_corner_yplus_labeled_batch3`
- `datasets/dji_action4_real_corner_yplus_labeled_all_v3`

总计：

- 当前可用真实角点标注约 `85` 张

### 7. 当前测试结果

已经生成多版视频测试结果。

当前重点结果：

- `runs/corner_resnet18_yplus_real_ft_v2/head_left_rgb_raw_yolo_conf025_pose.mp4`
- `runs/corner_resnet18_yplus_real_ft_v3/head_left_rgb_raw_yolo_conf025_pose.mp4`
- `runs/comparisons/head_left_rgb_raw_v2_vs_v3_side_by_side.mp4`

当前效果总结：

- 8 个点多数情况下已经落在相机本体范围内
- 部分帧已经能看出长方体结构
- 但仍存在点位不够准、编号偶尔互换、跨帧跳变的问题

## 三、当前卡点

当前最主要的卡点不是“代码没实现”，而是“效果还不够稳定”。

具体来说有三点：

1. 真实标注数量仍然不多
2. 部分真实标注可能存在编号不一致
3. 遮挡、模糊、快速运动时，角点容易跳变

所以当前瓶颈更偏向：

- **标注质量**
- **真实数据与合成数据之间的域差**
- **困难样本不足**

而不是单纯“网络完全没学会”。

## 四、当前结论

目前这条路线已经证明是可行的：

- 三维模型已经准备好
- 合成角点数据已经能自动渲染
- 角点 heatmap 网络已经实现
- 真实视频上已经能初步检测出 8 个角点

但距离“稳定、可信的结果”还有一段距离。

目前最合理的判断是：

- 老师流程已经基本走通
- 现在进入的是“提升质量”的阶段

## 五、接下来建议的优先级

### 第一优先级：清洗和复核真实标注

目标：

- 去掉明显不稳定样本
- 修正前后面、左右、上下编号不一致的问题

参考文档：

- `LABEL_QUALITY_CHECKLIST.md`

### 第二优先级：补少量高质量困难样本

优先补这几类：

- 镜头面清晰可见的图
- 背面清晰可见的图
- 当前视频里跳变最明显的帧

不建议优先补：

- 严重模糊图
- 大面积遮挡图
- 连镜头面都判断不清的图

### 第三优先级：基于清洗后的数据再微调一轮

目标：

- 看点位是否更准
- 看长方体关系是否更稳定
- 看编号互换是否减少

### 第四优先级：向老师确认任务预期

建议重点确认：

1. 是否允许使用少量真实人工角点标注做微调
2. 老师更看重流程完整性，还是更看重最终视频效果
3. 遮挡情况下是否允许按几何关系推断不可见角点
4. 8 个角点到底是按外接长方体定义，还是按更贴近真实外形定义

## 六、短期可执行计划

接下来最推荐的实际推进顺序：

1. 先复核现有真实标注
2. 清理掉明显有问题的样本
3. 再补一小批高质量困难帧
4. 用清洗后的数据重新微调
5. 重新导出测试视频
6. 结合结果决定是否继续补标，或者转入整理汇报

## 七、目前最重要的一句话

这个项目现在不是“从零开始”，也不是“完全做偏了”，而是：

**主流程已经跑通，当前重点是把真实视频上的 8 角点结果做得更稳定、更一致。**
