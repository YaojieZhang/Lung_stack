# Stack + scPhase 用于新辅助化免联合治疗配对单细胞数据的逐步操作指南

本文面向当前项目中的两个现有代码库：

- Stack：以 cell set 为基本输入单元，核心模型在 `stack/src/stack/models/core/base.py`，fine-tuning/ICL 后训练在 `stack/src/stack/models/finetune/model.py` 与 `stack/src/stack/finetune/lightning.py`。
- scPhase：以 patient/sample 作为 bag、单细胞作为 instance，MoE-MIL 聚合与 attention 解释在 `scPhase/scphase/model.py`、`scPhase/scphase/modules.py`、`scPhase/scphase/train_utils.py`。

你的任务可以拆成两个目标：

1. **虚拟治疗生成**：输入治疗前 scRNA-seq cell set，预测治疗后的单细胞表达谱。
2. **疗效预测和解释**：把同一患者生成/真实的细胞合并为 patient-level 表征，预测 MPR/NMPR，并回溯关键细胞和基因。

最重要的总体结论：**理论上可做，但必须把它定位为小样本迁移学习和机制假设生成，而不是从 7 例患者中训练一个完全可靠的新疗效预测模型。** 200 多个 cell sets 只能增加优化步数，不能增加独立患者数；所有模型选择和验证必须按 patient-level leave-one-patient-out 或 bootstrap 完成。

---

## 0. 推荐的最小可行架构

### 0.1 主架构

建议采用下面的保守架构：

```text
pre-treatment cells
    |
    |  cell-type matched set sampler, 256 cells/set
    v
Stack pretrained encoder + tabular attention
    |
    |  tiny treatment adapter / learned prompt, optional
    v
Stack NB decoder
    |
    +--> predicted post-treatment expression
    |
    +--> predicted post-treatment cell embeddings
            |
            v
      small scPhase-style MoE-MIL head
            |
            v
      patient-level MPR/NMPR prediction
```

尽量保留 Stack 原架构：

- 保留 `gene_reduction`、`TabularAttentionLayer`、`output_mlp` 和 NB decoder。
- 首轮不要重训 217M 参数；只训练新加的小模块、decoder 的最后一层或少量 adapter。
- scPhase 的 full model 可以作为 baseline；主模型中建议只复用其 **MoE-MIL 聚合思想**，不要直接从 7 个患者上训练完整 scPhase。

### 0.2 为什么不能直接“decoder 换成 MoE-MIL”

不建议把 Stack decoder 直接替换为 MoE-MIL，因为这会丢掉生成治疗后表达谱的目标。更合理的是：

- Stack decoder 继续负责 **single-cell expression generation**。
- MoE-MIL 作为 **parallel patient-level head** 或 downstream head，负责疗效分类。

即：

```text
Stack final_cell_embeddings
    ├── NB decoder -> generated post expression
    └── MoE-MIL head -> patient response logits
```

这样表达谱生成、疗效预测、可解释性三者都保留。

---

## 1. 小样本问题下的模型架构与微调参数

### 1.1 最佳策略：冻结主干 + 小模块微调

7 个配对患者不足以全量 fine-tune Stack。推荐分三档实验：

| 档位 | 可训练参数 | 适用目的 | 风险 |
|---|---|---|---|
| A. Frozen Stack | Stack 全冻结，只训练 treatment adapter + MoE-MIL head | 最稳的 baseline 和主结果候选 | 生成能力受限 |
| B. Decoder/adapter tuning | 冻结大部分 encoder，只训练 `output_mlp` 后半段、query/treatment prompt、MoE-MIL | 推荐主方案 | 中等过拟合 |
| C. Last-layer unfreeze | 解冻最后 1-2 个 `TabularAttentionLayer` + decoder + head | 作为增强实验 | 很容易过拟合 |

不要一开始就做：

- 全量 fine-tune Stack。
- 从零训练 Stack-like 模型。
- 在 set-level 随机划分 train/test。

### 1.2 Stack 参数建议

如果使用已有 pretrained checkpoint，尽量保持 checkpoint 的 gene list 和结构一致。若必须使用 Top 5k HVGs，优先把数据对齐到 Stack 的训练 gene list，不要轻易重建输入/输出层。

推荐初始设置：

| 参数 | 建议值 |
|---|---|
| `sample_size / n_cells` | 256，先与 Stack 论文和你的设想一致 |
| `n_kept_cell` | 64 或 96；如果 set size=256，先用 64 |
| `replacement_ratio` | 0.625-0.75；对应预测 160-192 个 query cells |
| `batch_size` | 1-4 sets/GPU |
| `accumulate_grad_batches` | 8-32，保证有效 batch 更稳定 |
| precision | `bf16-mixed` |
| gradient clipping | 1.0 |
| dropout | 0.05-0.15 for Stack adapter/decoder；0.2-0.4 for classifier head |
| optimizer | AdamW |
| backbone LR | 0 或 1e-6 到 3e-6 |
| decoder/adapter LR | 1e-5 到 3e-5 |
| MoE-MIL head LR | 1e-4 到 3e-4 |
| weight decay | 1e-4；bias/LayerNorm/embedding 可设 0 |
| epochs | 20-100，按 patient-level validation early stop |

scPhase 默认配置在 `scPhase/configs/config_NSCLC.json` 中使用 8 experts、256 hidden、5000 HVG 输入。对 7 个患者太大，建议缩小：

| 模块 | 默认思路 | 小样本建议 |
|---|---|---|
| MoE experts | 8 | 2-4 |
| hidden dim | 256 | 128 或直接用 Stack embedding dim 后接线性投影 |
| classifier dims | `[128,64]` | `[64,32]` |
| dropout | 0.1-0.2 | 0.3-0.5 |
| domain adaptation | 多 cohort 分类任务有用 | 7 例单队列先关闭，除非有明确 batch/domain |

### 1.3 Decoder 是否需要从头训练

不建议放弃 decoder 预训练参数，除非 gene list 或输出维度完全不兼容。

优先级：

1. **最佳**：使用 Stack checkpoint 对应 gene list，对你的数据做 gene alignment，缺失基因补 0，保留 decoder。
2. **可接受**：如果只用 Top 5k HVGs，初始化一个 HVG output adapter，从 Stack decoder 中按基因交集拷贝权重，缺失基因随机初始化。
3. **不推荐**：整套 decoder 从头训练。7 个患者即便有 120k cells，也会学到患者/批次噪声。

---

## 2. Dataset 设计

### 2.1 AnnData 元数据结构

建议将所有细胞放在一个 `.h5ad`，并至少包含：

```text
adata.X or adata.layers["counts"]     raw counts，用于 Stack NB loss
adata.layers["lognorm"]               log-normalized expression，用于 scPhase baseline
adata.obs["patient_id"]               患者 ID，7 个独立单位
adata.obs["sample_id"]                patient_id + timepoint
adata.obs["timepoint"]                pre / post
adata.obs["response"]                 MPR / NMPR
adata.obs["batch"]                    上机批次、测序批次或统一填 0
adata.obs["cell_type"]                细胞类型，建议先用中等粒度
adata.obs["treatment_regimen"]        化疗/ICB 方案，可选
adata.var_names                       gene symbols，需和 Stack genelist 对齐
```

注意：

- HVG 选择不要用响应标签。严格验证时，HVG 应在训练患者内选择，或直接使用 Stack 的固定 gene list。
- Stack 的 NB loss 应吃 counts；scPhase 原代码通常吃 5000 HVG log-normalized matrix。两个分支可以使用不同 layer，但 gene 顺序必须可追踪。

### 2.2 Cell type 处理

训练生成模型时，先只使用治疗前后共有的 cell types。

规则建议：

1. 每位患者、每个 timepoint、每个 cell type 至少保留一定细胞数，例如 `min_cells >= 64`。
2. 稀有类型合并到上级注释，例如 exhausted T / cytotoxic T 可先合并为 T/NK 大类，再在解释阶段细分。
3. 不要让 post-only cell type 进入监督生成 loss；它们可以作为后续探索，但不能用于成对训练。

### 2.3 Cell set 采样

每个 set 256 个细胞。关键不是随机抽 256 个，而是让 pre/post set 在 cell-type composition 上匹配。

推荐的 **simplex/Dirichlet 采样**：

1. 对患者 `p`，得到 pre/post 共有 cell types：`T_p`。
2. 计算每个 cell type 的 pre/post 最小比例或平均比例：`pi_p`。
3. 从 Dirichlet 分布采样 set composition：

   ```text
   c ~ Multinomial(256, q)
   q ~ Dirichlet(alpha * pi_p)
   ```

4. 对每个 cell type `t`，分别从 pre 和 post 中抽 `c_t` 个细胞，组成一对：

   ```text
   observed_features = pre_set[p, set_id, 256, genes]
   ground_truth_features = post_set[p, set_id, 256, genes]
   cell_type_ids = same order composition
   position_mask = unique cell mask
   ```

5. 对每位患者生成多个 set，但在训练/验证划分时，所有 set 必须随患者一起划分。

### 2.4 细胞配对方式

没有真实单细胞一一对应关系，所以不要假装 cell i pre 对 cell i post 是同一个细胞。推荐三种配对方式，从简单到复杂：

1. **cell-type stratified random pairing**：同一 cell type 内随机配对。最稳、最不容易引入额外假设。
2. **pseudobulk target**：不追求单细胞一一对应，只约束同一 cell type 的分布、均值和 DE direction。
3. **optimal transport pairing**：在 PCA/Stack frozen embedding 中做同 cell type 内 OT 匹配，用作增强实验。

主结果建议用 1 + 2，不要让结论依赖 OT。

### 2.5 Dataset 类的实现方向

当前 Stack fine-tuning dataset 返回：

```python
ground_truth_tensor, observed_tensor, cell_type_ids_tensor, position_mask, metadata
```

因此建议新增一个 `PairedTreatmentSetDataset`，输出同样格式，便于复用 `ICL_FinetunedModel.forward()`：

```text
ground_truth_tensor = post-treatment set
observed_tensor     = pre-treatment set, optionally plus treatment prompt
cell_type_ids       = matched cell type ids
position_mask       = mark real unique cells
metadata            = patient_id, response, set_id, timepoint, cell_ids
```

对 scPhase baseline，则保持其要求的 sample-level `.h5ad`：

```text
adata.obs["sample_id"] = patient_id 或 generated_patient_id
adata.obs["phenotype"] = MPR/NMPR
adata.obs["batch"] = batch
```

---

## 3. 损失函数设计

### 3.1 生成损失

Stack 原始 core 使用 masked NB reconstruction + Sliced Wasserstein latent regularization。fine-tuning 中还有 MMD 和 cls loss。你的任务建议使用：

```text
L_gen =
  lambda_nb      * L_NB(post_counts, pred_post_counts)
+ lambda_mmd     * L_MMD_by_cell_type(pred_post, true_post)
+ lambda_delta   * L_delta_pseudobulk(pred_post - pre, true_post - pre)
+ lambda_sw      * L_SW(latent)
+ lambda_self    * L_self_reconstruction
```

各项含义：

- `L_NB`：负二项分布 NLL，保留 Stack decoder 对 count noise 的建模。
- `L_MMD_by_cell_type`：同一患者、同一 cell type 内，预测 post 分布和真实 post 分布的 MMD/energy distance。
- `L_delta_pseudobulk`：关键项。约束治疗前后变化方向，而不强迫单细胞一一对应。
- `L_SW`：保留 Stack latent distribution 正则，防止 embedding 崩。
- `L_self_reconstruction`：pre->pre、post->post 的轻量自重构，防止模型把所有输入都推向平均 post。

初始权重建议：

```text
lambda_nb    = 1.0
lambda_mmd   = 0.5
lambda_delta = 0.5
lambda_sw    = 0.05-0.1
lambda_self  = 0.1
```

训练后期可以增大 `lambda_delta`，因为疗效相关信号更多体现在治疗诱导变化，而不是绝对表达。

### 3.2 疗效预测损失

MPR/NMPR 是 patient-level 分类：

```text
L_cls = class-balanced CE 或 focal loss
```

推荐：

- 如果类别接近平衡：weighted cross entropy。
- 如果 MPR/NMPR 明显不平衡：focal loss 或 class-balanced focal。
- 不建议直接让 200+ sets 都拥有独立标签后平均训练，这会把 patient label 复制成很多伪样本。正确做法是 set-level logits 先聚合到 patient，再算 patient-level loss。

patient-level 聚合：

```text
cell embeddings -> set embeddings -> patient embedding -> response logits
```

或：

```text
all generated cells from one patient -> MoE-MIL -> patient logits
```

如果显存受限，可每个 patient 抽多个 sets，训练时对同一 patient 的 set logits 做 mean/logsumexp pooling。

### 3.3 总损失

推荐两阶段或三阶段，不要一开始就全损失联合：

```text
Stage A: L = L_gen
Stage B: L = L_cls, Stack frozen
Stage C: L = L_gen + lambda_cls * L_cls + lambda_consistency * L_delta
```

联合微调初始权重：

```text
lambda_cls = 0.1-0.3
```

如果发现分类 loss 使生成质量变差，停止联合微调，保留两阶段结果。

---

## 4. 小样本微调策略

### 4.1 数据划分

唯一合理的主验证：

```text
Leave-One-Patient-Out CV
```

每一折：

```text
train patients: 6
test patient: 1
optional validation: 从 train patients 中再留 1 个，或固定超参不做内层调参
```

不要使用：

- set-level random split。
- cell-level split。
- 同一患者的部分 set 训练、部分 set 测试。

### 4.2 训练日程

建议主流程：

1. **Stage 0：Stack frozen zero-shot**
   - 不训练，直接评估 pre->post 生成的 pseudobulk delta 和 cell-type delta。
   - 用作所有后续训练是否真的有效的参照。

2. **Stage 1：只训练 treatment adapter/decoder tail**
   - 冻结 Stack encoder 和 attention blocks。
   - 训练 20-50 epochs。
   - early stopping 指标：held-out train-validation patient 的 `delta_corr`、`MMD`、`masked_corr`。

3. **Stage 2：训练 MoE-MIL response head**
   - 使用真实 post、预测 post、pre 三种输入各训练一个 head。
   - Stack frozen。
   - 输出 patient-level MPR/NMPR。

4. **Stage 3：可选联合微调**
   - 解冻最后 1-2 个 Stack attention layers。
   - backbone LR 降到 head LR 的 1/20 到 1/100。
   - 只跑少量 epochs，例如 5-20。

5. **Stage 4：解释和 ablation**
   - 计算 attention/IG。
   - 做 remove-high-attention-cells、mask-top-genes 验证解释是否影响 logits。

### 4.3 防过拟合检查

每折必须记录：

- train vs validation/test 的 `delta_pseudobulk_corr`。
- 每个 patient 的 prediction probability，而不是只报平均 AUC。
- label permutation test：打乱 MPR/NMPR 后重训/重评 50-100 次，主模型应显著优于置换。
- seed sensitivity：至少 5 个 random seeds。
- high-attention cells 是否集中到某一个患者/批次；若是，说明模型可能学了 batch。

---

## 5. 建议新增模块

### 5.1 `PairedTreatmentSetDataset`

目的：按患者、cell type、timepoint 生成 paired cell sets。

输入：

```text
adata
patient_id_col
timepoint_col
response_col
cell_type_col
gene_list
n_cells=256
sampler="dirichlet" or "fixed_composition"
```

输出保持 Stack fine-tuning 兼容：

```python
ground_truth_tensor, observed_tensor, cell_type_ids_tensor, position_mask, metadata
```

### 5.2 `TreatmentPromptAdapter`

这是最小的新架构模块，用来表示“化免联合治疗”这个条件。

两种实现：

1. **token-space adapter**：对 Stack tokens 加一个可学习 treatment embedding。
2. **pseudo-cell prompt**：学习少量 pseudo-cells/pseudo-tokens，拼到 cell set 前面。

小样本推荐第一种，因为参数更少：

```text
tokens = tokens + treatment_embedding
```

如果有不同化疗方案，可以给 regimen 一个 embedding，但 7 例患者不足以学习复杂 regimen effect。先把治疗当作单一条件，regimen 只作为协变量或敏感性分析。

### 5.3 `StackMoEMILHead`

复用 scPhase 的 `MoEMILAggregation` 思想，但输入最好是 Stack embedding 或 generated delta，而不是直接 5000 基因。

推荐输入特征：

```text
cell_feature = concat(
    predicted_post_embedding,
    pre_embedding,
    predicted_post_embedding - pre_embedding,
    optional cell_type_embedding
)
```

然后：

```text
MoE-MIL attention -> patient embedding -> small classifier
```

### 5.4 `InterpretabilityBridge`

必须保存从 patient 到 set 到 cell 到 gene 的映射：

```text
patient_id
set_id
original_cell_barcode
generated_cell_index
cell_type
attention_weight
gene_attribution
treatment_delta
```

解释输出：

- cell-level attention from MoE-MIL。
- Stack inter-cell attention from `return_attn=True` 的 tabular attention。
- gene-level IG / gradient x input。
- cell-type-level keyness score。
- top gene modules and pathway enrichment。

---

## 6. Baseline 设计

为了让审稿人相信模型不是“自娱自乐”，baseline 必须强且贴近任务。

### 6.1 生成任务 baselines

1. **Identity baseline**
   - 预测 post = pre。
   - 如果模型不能超过它，说明没有学到治疗 effect。

2. **Mean delta by cell type**
   - 从训练患者估计每个 cell type 的平均治疗变化：

   ```text
   pred_post = pre + mean_train_delta[cell_type]
   ```

   这是非常强的小样本 baseline。

3. **Nearest-patient delta**
   - 用 pre pseudobulk 找最相近训练患者，把该患者治疗 delta 转移到测试患者。

4. **OT / mapping baseline**
   - 同 cell type 内用 optimal transport 匹配 pre/post distribution。

5. **Stack frozen zero-shot**
   - 不训练 Stack，只用 pretrained/fine-tuned checkpoint 的 ICL generation。

6. **Conditional VAE/scGen-like baseline**
   - 如果实现成本允许，用 cell type + treatment condition 作为条件，预测 post。
   - 小样本下未必强，但审稿人熟悉。

### 6.2 疗效预测 baselines

1. **Pre-treatment pseudobulk elastic net / ridge logistic**
   - 每个 cell type pseudobulk + L2 logistic。

2. **Cell type abundance logistic**
   - 只用治疗前 cell type proportion。

3. **scPhase original**
   - 输入 pre cells 或真实 post cells。
   - patient-level LOPO。

4. **Frozen Stack embedding + mean pooling**
   - Stack embedding 不训练，patient mean pooling + logistic。

5. **Frozen Stack embedding + simple attention MIL**
   - 不用 MoE，只用 gated attention。

6. **Full proposed model ablations**
   - no treatment adapter。
   - no generation loss。
   - no MoE，只 mean pooling。
   - no cell-type matched sampler。
   - frozen vs last-layer unfreeze。

### 6.3 评价指标

生成任务：

- cell-type pseudobulk Pearson/Spearman correlation。
- delta correlation：`corr(pred_post - pre, true_post - pre)`。
- DE direction accuracy。
- top-k DE overlap。
- MMD/SW distance by cell type。

疗效预测：

- LOPO balanced accuracy。
- ROC-AUC 只能谨慎报告，因为 n=7 太小。
- PR-AUC、F1、Brier score。
- 每个患者预测概率和置信区间。
- permutation p-value。

---

## 7. 训练步骤流程

### Step 1：数据 QC 和统一基因空间

1. 读取原始 `.h5ad`。
2. 保留 raw counts 到 `adata.layers["counts"]`。
3. 标准化/log1p 存到 `adata.layers["lognorm"]`。
4. 统一 gene symbols 大小写和别名。
5. 与 Stack genelist 对齐，缺失基因补 0。
6. 标注并检查：

```text
patient_id, sample_id, timepoint, response, batch, cell_type
```

### Step 2：探索性统计

对每位患者输出：

- pre/post cell number。
- pre/post 共有 cell types。
- 每个 cell type 的细胞数。
- MPR/NMPR 标签。
- treatment regimen。

如果某个 cell type 在多数患者中 post 缺失，不要纳入首轮监督生成。

### Step 3：构建 paired cell sets

每位患者：

1. 取共有 cell types。
2. 用 Dirichlet/simplex 抽 composition。
3. 从 pre/post 分别按 composition 抽 256 cells。
4. 保存 set metadata。

建议每位患者先生成 30-100 个 sets，而不是无限增强。增强太多只会让模型更自信地记住患者。

### Step 4：训练 Stack 生成模型

每个 LOPO fold：

1. test patient 完全留出。
2. train patients 生成 train sets。
3. validation 若可行，从 train patients 中留 1 个 patient。
4. 加载 pretrained Stack checkpoint。
5. 冻结 encoder/attention，先训练 adapter/decoder。
6. 保存最佳 checkpoint。

监控：

```text
val L_NB
val MMD by cell type
val delta_corr
val DE_direction_accuracy
```

### Step 5：生成 held-out patient 的虚拟 post cells

对 test patient：

1. 输入 pre cells。
2. 不使用该患者真实 post cells 作为 prompt，否则是信息泄漏。
3. 使用训练患者形成的 global treatment prompt，或 learned treatment adapter。
4. 输出 generated post `.h5ad`。
5. 与真实 post 只在 evaluation 阶段比较。

### Step 6：训练/评估 patient-level 疗效预测

建议三种输入都做：

1. pre-only：治疗前真实细胞。
2. generated-post：Stack 预测治疗后细胞。
3. true-post oracle：真实治疗后细胞，上限参考，不能作为临床预测主结果。

每种输入用同一套 patient-level split。

如果只 7 个患者，主结果更应该是：

```text
每个 held-out patient 的预测概率 + permutation test + baseline 对照
```

而不是只报一个 AUC。

### Step 7：解释性分析

对每个 fold 的 held-out patient：

1. 取 MoE-MIL attention，得到 high-impact cells。
2. 计算 IG/gradient x input，得到 high-impact genes。
3. 对 high-impact cells 做 cell type 汇总。
4. 对 high-impact genes 做 pathway enrichment。
5. 做解释验证：

```text
remove top-attention 5% cells -> response logit drop?
mask top-attribution genes -> response logit drop?
replace high-impact cell type with average cells -> response logit drop?
```

最终输出：

```text
patient_id
predicted_response
key_cell_types
key_cells
key_genes
gene_programs
evidence from ablation
```

---

## 8. 对你当前 7 个问题的直接回答

### Q1. 小样本下最佳模型架构和训练注意事项

主推：**Stack pretrained backbone + tiny treatment adapter + NB decoder + small MoE-MIL patient head**。冻结大部分 Stack，只训练少量参数。所有 split 按 patient-level。不要把 200+ sets 当作独立样本。

### Q2. 如何设计 dataset

使用 `PairedTreatmentSetDataset`：

```text
pre set  -> observed_features
post set -> ground_truth_features
same cell-type composition
same gene order
patient-level split
```

同时准备 scPhase 格式 `.h5ad`，用于 baseline 和 MoE-MIL 解释。

### Q3. 如何设计损失函数

使用多目标：

```text
NB count loss
+ cell-type MMD/SW distribution loss
+ pseudobulk treatment delta loss
+ patient-level class-balanced CE/focal loss
+ self-reconstruction regularization
```

先分阶段训练，再尝试联合损失。

### Q4. 小样本微调策略

冻结主干，先训练 adapter/decoder/head；必要时只解冻最后 1-2 层。学习率分层：backbone 1e-6，decoder/adapter 1e-5，classifier 1e-4。使用 LOPO、早停、多 seed、label permutation。

### Q5. 添加哪些新模块

最少新增四个：

1. `PairedTreatmentSetDataset`
2. `TreatmentPromptAdapter`
3. `StackMoEMILHead`
4. `InterpretabilityBridge`

### Q6. 最贴合需求的 baseline

必须包括：

- identity pre=post。
- training-patient mean delta by cell type。
- nearest-patient delta。
- scPhase original pre-only。
- frozen Stack embedding + logistic/MIL。
- proposed model ablations。

### Q7. 训练流程

流程是：

```text
QC/gene alignment
-> paired set sampling
-> Stack frozen zero-shot
-> Stack adapter/decoder SFT
-> generated post expression
-> MoE-MIL patient classifier
-> LOPO evaluation
-> attention + IG + ablation interpretation
-> baseline comparison
```

---

## 9. 最容易踩的坑

1. **set-level 泄漏**：同一患者的 sets 同时出现在 train/test，会严重高估性能。
2. **post prompt 泄漏**：预测 held-out 患者时不能使用该患者 post cells 作为 prompt。
3. **把细胞数当样本量**：120k cells 不等于 120k labels；疗效标签只有 7 个独立样本。
4. **全量 fine-tuning**：217M 参数会记住患者特异噪声。
5. **解释只看 attention**：attention 需要 IG 和 removal ablation 支撑。
6. **过度细分 cell type**：小样本下先用中等粒度，解释阶段再细分。
7. **HVG/gene list 泄漏或不一致**：Stack 和 scPhase 的输入 gene space 必须明确记录。

---

## 10. 第一轮实验的推荐最小配置

如果只跑第一版，我建议：

```text
n_cells = 256
n_kept_cell = 64
sets_per_patient = 50
Stack = frozen except treatment adapter + decoder output tail
MoE experts = 2
patient head hidden = 64
dropout = 0.3
generation epochs = 30
classifier epochs = 100 with early stopping
LOPO folds = 7
seeds = 5
baselines = identity, mean-delta, scPhase pre-only, frozen Stack embedding
```

第一轮目标不是追求最高 AUC，而是回答三个问题：

1. 生成 post 是否优于 identity/mean-delta baseline？
2. generated-post 是否比 pre-only 更有疗效预测信息？
3. 高重要性细胞/基因是否能通过 removal 或 masking 改变模型输出？

如果这三点成立，后续再考虑引入更多公共数据、regimen embedding、更复杂的 perturbation module 或联合训练。
