# ComfyUI-LCS

基于**潜在颜色子空间**（Latent Color Subspace）的免训练颜色控制，以及基于发现的**锐度子空间**的锐度控制。

> **注意：** 本项目为非官方社区实现。官方代码见 [ExplainableML/LCS](https://github.com/ExplainableML/LCS)。

基于论文 ["The Latent Color Subspace"](https://arxiv.org/abs/2603.12261v1)（ICML 2026）：扩散模型潜在 patch 空间中的颜色完全存在于一个 **3 维子空间**（PCA 捕获 100% 颜色方差），剩余 61 维编码结构与细节，与颜色正交。

本插件在扩散采样过程中直接操作 3D LCS 控制颜色——无需训练、无需 LoRA、无需后处理。

> [English README](README.md)

## LCS 与传统后处理调色的区别

LCS 在扩散采样**过程中**操作，而非生成之后——这是与传统调色（Photoshop、滤镜等）的根本区别。

| | 传统后处理 | LCS |
|---|---|---|
| **时机** | VAE 解码后，像素空间 | 采样过程中，潜在空间 |
| **机制** | 对成品图像施加颜色滤镜 | 在生成中途修改 3D 颜色子空间 |
| **模型感知** | 无——结构已定型 | 模型在后续步骤中自适应颜色偏移 |
| **效果** | 颜色容易显得"涂上去的" | 颜色与内容自然融合 |

例：想要暖橙色日落，后处理会给全图叠橙色（阴影和肤色变脏），而 LCS 在采样早期推动颜色子空间，模型生成的云层、光照、反射与暖色调**内在一致**。

核心发现：颜色与结构在潜在 patch 空间中**正交**——可以单独控制颜色而不干扰结构。

## 已测试模型

| 模型 | 状态 |
|------|------|
| FLUX | 已测试 |
| FLUX2.klein | 已测试 |
| z-image | 已测试 |
| z-image-turbo | 已测试 |
| Wan (qwen-image) | 已测试 |
| LTX2.3 | 已测试 |

LCS 按 VAE 校准，理论上适用于任何使用兼容 VAE 架构的模型。欢迎反馈其他模型的测试结果。

## 功能

- **颜色引导** — 将颜色推向任意目标色
- **批量多色** — 为批次中每张图像指定不同颜色
- **色调调整** — 对比度、亮度、饱和度、色温，支持一键预设
- **颜色锚定** — 零配置颜色漂移校正：自锚定、参考图锚定、空间平滑，支持全自动模式
- **锐度控制** — 在生成过程中增强或减弱锐度，基于发现的锐度子空间（PC1 解释 ~97% 方差）
- **局部控制** — 可选遮罩，实现区域性变化
- **潜在颜色预览** — 无需 VAE 解码即可可视化颜色结构
- **步骤观察器** — 保存每步颜色预览，用于调试

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/facok/ComfyUI-LCS.git
```

依赖（通常 ComfyUI 已自带）：

```bash
pip install einops safetensors
```

## 快速开始

### 基本颜色控制

```
LCS Load Data → LCS Color Intervene → KSampler
                       ↑
                  （选择颜色）
```

1. **LCS Load Data** — 连接 VAE（首次运行自动校准）
2. **LCS Color Intervene** — 连接 MODEL 和 LCS_DATA，选择目标颜色
3. 将输出 MODEL 连接到 KSampler

### 色调调整

```
LCS Load Data → LCS Tone Adjust → KSampler
```

1. **LCS Load Data** → **LCS Tone Adjust**
2. 选择预设（如 "Cinematic"）或手动调整滑条

![3d3c82eb0e89ed1608e40ac7a8cc3408](https://github.com/user-attachments/assets/62868e2d-0275-4801-a9bd-606bfea3ce2f)
![42541357](https://github.com/user-attachments/assets/fe22f09e-98ac-4281-ae40-f58232c7700f)
### 锐度控制

```
LCS Load Data ──→ LCS Sharpness Calibrate → LCS Sharpness Intervene → KSampler
                        ↑ lcs_data
```

1. **LCS Sharpness Calibrate** — 连接 VAE（首次运行自动校准并缓存）。可选连接 `lcs_data`（来自 LCS Load Data），确保锐度编辑不影响颜色。
2. **LCS Sharpness Intervene** — 连接 MODEL 和 SHARPNESS_DATA，设置强度
   - 正值 → 更锐利
   - 负值 → 更模糊
   - 0 → 无变化
![89814728](https://github.com/user-attachments/assets/62f036e9-0bea-4cc0-9220-af4c2fb8fa76)

### 批量多色生成

```
LCS Load Data → LCS Color Batch → KSampler
                      ↓
                  batch_size → EmptyLatentImage
```

输入逗号分隔的十六进制颜色（如 `#FF0000,#00FF00,#0000FF`），每个颜色对应一个批次项。

### 颜色锚定（零配置漂移校正）

```
LCS Load Data → LCS Color Anchor → KSampler
```

1. **LCS Load Data** → **LCS Color Anchor** — 连接 MODEL 和 LCS_DATA
2. 模式设为 **auto**（默认），intensity 保持默认值
3. 将输出 MODEL 连接到 KSampler

完成。在 `auto` 模式下，节点根据连接的可选输入自动选择校正策略：

| 已连接输入 | 解析模式 | 行为 |
|---|---|---|
| 无 | self_anchor | 在早期学习图像的颜色规律，然后防止突然的颜色偏移 |
| reference_image + vae | reference | 让生成的颜色贴近你的参考图 |
| mask（无参考图） | smooth | 平滑颜色接缝（很适合修复/补绘） |

intensity 也会根据实测漂移自动推导——无需手动调参。

> **手动模式：** 如果需要完全控制，可以将模式设为 `smooth`、`reference` 或 `self_anchor`，并手动调节 `intensity` 滑条（0–1）。auto 模式适合零配置「开箱即用」场景。

## 节点一览

### 校准

| 节点 | 说明 |
|------|------|
| **LCS Load Data** | 自动校准并按 VAE 缓存 LCS 颜色数据。通过 VAE 权重指纹自动管理缓存。 |
| **LCS Sharpness Calibrate** | 通过模糊刺激 PCA 发现锐度子空间。可选连接 `lcs_data` 使锐度正交于颜色。 |

每个 VAE 只需校准一次，结果自动缓存，后续运行瞬时加载。

### 干预

| 节点 | 说明 |
|------|------|
| **LCS Color Intervene** | 将颜色引导至目标色。支持 Type I（LCS 平移）、Type II（HSL 偏移）或插值模式。 |
| **LCS Color Batch** | 每个批次项施加不同目标颜色。输出 `batch_size` 可连接 EmptyLatentImage。 |
| **LCS Tone Adjust** | 对比度、亮度、饱和度、色温调整。预设下拉菜单，滑条实时同步。 |
| **LCS Color Anchor** | 采样过程中校正颜色漂移。auto 模式根据连接输入自动推断策略和强度。 |
| **LCS Sharpness Intervene** | 在生成过程中控制锐度。正值 = 更锐利，负值 = 更模糊。 |

### 观察

| 节点 | 说明 |
|------|------|
| **LCS Preview Colors** | 将潜在颜色解码为 RGB 预览图，无需 VAE 解码。 |
| **LCS Step Observer** | 将每步颜色预览 PNG 保存至 ComfyUI 临时目录。 |

## 干预模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **interpolated**（默认） | 以 sigma 为权重混合 Type I 和 Type II | 通用场景 |
| **type_i** | 3D LCS 空间中的直接平移 | 强烈的全局颜色偏移 |
| **type_ii** | 通过双锥几何进行逐 patch HSL 插值 | 精确的局部颜色控制 |

## 关键参数

### 颜色干预
- **strength**（0.0–2.0）：干预强度。1.0 = 完整干预，0.0 = 无干预。
- **start_step / end_step**：干预步骤范围。论文最优：50 步中的第 8–10 步。
- **mask**：可选。下采样至 patch 网格分辨率，用于局部控制。

### 锐度干预
- **strength**（-5.0–5.0）：正值 = 更锐利，负值 = 更模糊，0 = 无变化。
- **start_step / end_step**：干预步骤范围（默认 5–15）。
- **mask**：可选。用于局部锐度控制。

> **步数蒸馏模型提示**：对于步数蒸馏模型（如 z-image-turbo），总步数很少，干预应从更早的步骤开始——甚至可以从第 0 步就开始干预。

### 颜色锚定

扩散模型在采样过程中有时会出现意想不到的颜色偏移——蓝天突然变紫，或者修复/补绘后留下明显的颜色接缝。颜色锚定节点在图像生成过程中监控和修正这些问题。

**模式：**

| 模式 | 功能 | 适用场景 |
|------|------|----------|
| **auto**（默认） | 根据你连接的输入自动选最合适的策略 | 不想调参，开箱即用 |
| **self_anchor** | 在早期步骤观察颜色变化规律，在后续步骤防止突然的颜色跳变 | 通用颜色稳定，不需要参考图 |
| **reference** | 让生成图像的颜色贴近你提供的参考图 | 「我想要这张照片的配色风格」 |
| **smooth** | 平滑区域之间的突兀颜色边界 | 修复/补绘后消除接缝 |

**auto 模式如何自动选择：**

1. **用哪种策略？** 看你连了什么：
   - 连了参考图 + VAE → 用 `reference`
   - 连了遮罩（没有参考图）→ 用 `smooth`
   - 什么额外输入都没连 → 用 `self_anchor`
2. **修正多强？** 节点会测量实际的颜色漂移幅度，据此自动设置校正强度。漂移大 → 修正更强；漂移小 → 轻轻一碰。范围是 0.15–0.6，既不会矫枉过正，也不会毫无作用。

**采样过程中发生了什么：**

节点在每个采样步都会运行，但不会每步都干预。它自动判断哪些步骤适合校正：

1. **早期步骤**（图像基本是噪声）— 太早修正颜色会产生伪影，跳过。在 self_anchor 模式下，节点利用这些步骤*学习*图像的颜色规律。
2. **中间步骤**（图像逐渐成形）— 最佳校正时机。节点在这里施加校正，平滑地渐入渐出，避免突变。
3. **后期步骤**（精细细节）— 校正会干扰细节，跳过。

只修改颜色——结构、纹理、细节始终不受影响。

**参数：**

- **mode**：`auto`、`smooth`、`reference` 或 `self_anchor`
- **intensity**（0.0–1.0）：校正强度。auto 模式下自动决定。设为 0 可完全禁用此节点。
- **vae**（可选）：reference 模式需要用它来编码参考图
- **reference_image**（可选）：你想匹配其颜色的参考图
- **mask**（可选）：只在遮罩区域内校正颜色

## 色调预设

选择预设后滑条实时更新。可在预设基础上微调。选择 **Custom** 可完全手动设置。

| 预设 | 对比度 | 亮度 | 饱和度 | 色温 |
|------|--------|------|--------|------|
| Base | 1.0 | 0.0 | 1.0 | 0.0 |
| Cinematic | 1.20 | -0.05 | 0.90 | 0.05 |
| HDR | 1.40 | 0.0 | 1.20 | 0.0 |
| Vivid | 1.10 | 0.0 | 1.50 | 0.0 |
| Dramatic | 1.50 | -0.10 | 0.85 | 0.0 |
| Low Key | 1.30 | -0.20 | 0.80 | 0.0 |
| High Key | 0.80 | 0.20 | 0.90 | 0.0 |
| Warm | 1.05 | 0.03 | 1.10 | 0.30 |
| Cool | 1.05 | 0.0 | 1.05 | -0.30 |
| Desaturated | 1.0 | 0.0 | 0.40 | 0.0 |

## 工作原理

### 颜色（LCS）

1. **投影** — 将去噪预测转换到 64D patch 空间，投影到 3D LCS 基底
2. **分解** — 将 3D 颜色坐标与 61D 结构残差分离
3. **归一化** — 使用学习的 alpha/beta 统计量变换至参考时间步（t=50）
4. **操作** — 在 3D LCS 中偏移颜色、调整色调或进行其他变换
5. **重建** — 反归一化，加回保留的 61D 残差，转换回潜在空间

61D 残差（结构、纹理、细节）始终不被修改——只有 3D 颜色子空间会被改变。

### 锐度

锐度存在于与颜色正交的独立子空间中：

1. **校准** — 生成灰度噪声图像，应用多级高斯模糊，VAE 编码后对去除颜色分量的 patch 向量做 PCA。PC1 捕获 ~97% 的锐度方差。
2. **干预** — 在每个 patch 上沿 `strength * pc1_direction` 方向添加偏移。由于 pc1_direction 与颜色正交（校准时已移除 LCS 分量）且无直流分量（PCA 前做了逐向量零均值化），因此只改变空间频率内容，不影响颜色或亮度。

### 颜色锚定

颜色锚定的作用是稳定颜色，而不是把颜色推向某个特定目标——它防止模型已经在生成的颜色发生偏移：

1. **判断何时介入** — 节点检查每个采样步：图像还是一片噪声（太早）、正在成形（适合校正）、还是快完成了（太晚）？只在安全的中间窗口进行校正。
2. **学习颜色规律**（self_anchor）— 在早期噪声较大的步骤中，节点观察每个区域的颜色与邻居之间的关系，建立一个动态平均值。比起追踪绝对颜色值，这种「相对关系」更可靠，因为绝对颜色在图像成形过程中本来就会自然变化。
3. **测量漂移** — 在第一个校正步，节点测量颜色实际漂移了多少（根据模式不同：步间跳变幅度、与参考图的差距、或空间粗糙程度）。这决定了 auto 模式下的校正强度。
4. **温和地修正** — 校正平滑地渐入渐出（不会突变）。每种模式的修正方式不同：self_anchor 修复偏离已学规律的区域，reference 拉近与参考图的颜色，smooth 模糊掉尖锐的颜色边界。
5. **保留其他一切** — 与所有 LCS 操作一样，只修改 3D 颜色坐标，结构、纹理、细节完全不受影响。

## 文件结构

```
ComfyUI-LCS/
├── __init__.py           # 入口（V3 + V2 兼容）
├── requirements.txt
├── core/
│   ├── adaptive.py       # 自适应调度（阶段、包络、漂移估计）
│   ├── bilateral.py      # LCS 颜色平滑的双边滤波
│   ├── calibration.py    # PCA 校准流程（颜色）
│   ├── color_space.py    # 双锥 LCS ↔ HSL 映射
│   ├── defaults.py       # 论文中的 Alpha/beta 表
│   ├── lcs_data.py       # LCSData 数据类
│   ├── patchify.py       # Patch ↔ 潜在空间转换
│   ├── relationships.py  # 局部颜色关系分析与异常检测
│   ├── sampling.py       # 共享常量和步骤工具
│   ├── sharpness.py      # 锐度子空间校准
│   └── timestep.py       # Sigma/时间步工具
├── nodes/
│   ├── anchor.py         # LCSColorAnchor（自适应颜色漂移校正）
│   ├── calibrate.py      # LCSLoadData（自动校准 + 缓存）
│   ├── intervene.py      # LCSColorIntervene, LCSColorBatch, LCSToneAdjust
│   ├── observe.py        # LCSPreviewColors, LCSStepObserver
│   └── sharpen.py        # LCSSharpnessCalibrate, LCSSharpnessIntervene
├── data/                 # 缓存的校准文件
└── web/js/
    └── tone_preset.js    # 前端预设同步
```

## 更新日志

### 2026-03-21
- **颜色锚定：auto 模式** — 新增 `auto` 模式，根据连接的输入自动推断校正策略（self_anchor / reference / smooth），并根据实测漂移推导强度。零配置使用。
- **颜色锚定：自适应调度** — 阶段分配（observe/correct/skip）和强度包络在运行时从 sigma 调度表推导。

### 2026-03-20
- **锐度控制** — 通过模糊刺激 PCA 发现锐度子空间。新增 `LCS Sharpness Calibrate` + `LCS Sharpness Intervene` 节点。PC1 解释 ~97% 方差，与颜色正交。
- **颜色正交锐度** — 可选连接 `lcs_data`，在锐度校准时移除颜色分量，防止颜色偏移。

### 2026-03-19
- **视频 VAE 支持（Wan）** — 在 patchify/unpatchify 中处理 5D 视频潜在表示。视频 VAE 自动回退到逐帧编码。
- **LTXV 兼容** — patchify 中填充奇数空间维度，处理 3D 张量，不兼容格式时优雅跳过。
- **FLUX2 支持** — unpatchify 自动检测 128 通道潜在表示。
- **通用潜在格式** — 使用模型的 `latent_format` 进行空间转换，不再硬编码 FLUX 常量。

### 2026-03-18
- **色调调整** — `LCS Tone Adjust` 节点，支持对比度、亮度、饱和度、色温滑条。10 个预设，前端实时同步。
- **色温控制** — 沿 LCS 蓝-黄轴的暖/冷偏移。
- **双锥 HSL 几何** — 通过双锥 LCS-to-HSL 映射实现正确的 Type II 干预。

### 2026-03-17
- **首次发布** — 颜色引导（Type I + Type II + 插值模式）、批量多色、局部遮罩控制、潜在颜色预览、步骤观察器。按 VAE 自动校准并缓存。

## 引用

官方仓库：[ExplainableML/LCS](https://github.com/ExplainableML/LCS)

```bibtex
@article{pach2026latentcolorsubspace,
  title={The Latent Color Subspace: Emergent Order in High-Dimensional Chaos},
  author={Mateusz Pach and Jessica Bader and Quentin Bouniot and Serge Belongie and Zeynep Akata},
  journal={arxiv},
  year={2026}
}
```

## 致谢

感谢 Mateusz Pach、Jessica Bader、Quentin Bouniot、Serge Belongie 和 Zeynep Akata，他们的研究使免训练颜色控制成为可能。

## 许可证

MIT
