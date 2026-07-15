当前一共改了 3 个文件：

```text
configs/finetuning/ft_parsecg.yaml
src/stack/finetune/utils.py
src/stack/data/finetuning/datasets.py
```

总 diff：`209 insertions, 61 deletions`。

**1. 配置层**
[ft_parsecg.yaml](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/configs/finetuning/ft_parsecg.yaml:1)

已完成：

- 把 `dataset_configs` 从原来的 `human` 数据集改成 `paired` 数据集：
  ```yaml
  paired:/data/home/zhangyaojie/data/NG/neoadjuvant_paired:Patient:Timepoint:cell_type:Pre-biopsy:Post-surgery:false:gene_symbols
  ```
- 这个字符串现在携带 8 类信息：
  `dataset_type, path, patient_col, timepoint_col, cell_type_col, pre_condition, post_condition, filter_organism, gene_name_col`
- 修复了之前多出来的一个双引号，YAML 现在可以正常解析。
- 同步改了 checkpoint、genelist、save_dir、cache_file 为你当前服务器路径。

**2. 配置解析层**
[utils.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/finetune/utils.py:15) 的 `parse_dataset_configs`

已完成：

- 新增 `paired` 类型解析分支。
- 支持格式：
  ```text
  paired:path:patient_col:timepoint_col:cell_type_col:pre_condition:post_condition:filter_organism:gene_name_col
  ```
- 解析后会构造：
  ```python
  DatasetConfig(
      type="paired",
      patient_col="Patient",
      timepoint_col="Timepoint",
      cell_type_col="cell_type",
      pre_condition="Pre-biopsy",
      post_condition="Post-surgery",
      filter_organism=False,
      gene_name_col="gene_symbols",
  )
  ```

**3. DatasetConfig 结构层**
[datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:34) 的 `class DatasetConfig`

已完成：

- `type` 支持从 `human/drug` 扩展为 `human/drug/paired`。
- 新增 paired 字段：
  ```python
  patient_col
  timepoint_col
  pre_condition
  post_condition
  ```
- `__post_init__()` 增加 paired 必需字段校验。
- `group_col` 对 paired 返回 `patient_col`，即按 Patient 分组。
- `identity_col` 对 paired 返回 `cell_type_col`，即按细胞类型匹配。

**4. Metadata 构建层**
[datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:268) 的 `MultiDatasetMetadataCache._build_all_metadata`

已完成 paired 适配：

- 新增 metadata 数组：
  ```python
  self.patient_ids
  self.timepoints
  ```
- 对 paired 数据要求 obs 中必须存在：
  ```text
  Patient
  Timepoint
  cell_type
  ```
- 新增统一读取 obs 列的内部 helper：支持 h5ad categorical 的 `categories/codes` 和普通 dataset。
- paired 数据只保留指定 timepoint：
  ```text
  Pre-biopsy
  Post-surgery
  ```
- 对 paired：
  ```python
  group_ids = Patient
  identities = cell_type
  conditions = Timepoint
  patient_ids = Patient
  timepoints = Timepoint
  ```
- `file_info["organism_mask"]` 现在保存最终 `valid_mask`，也就是 organism filter + paired timepoint filter 后的 mask。
- `group_mapping` 中额外保存 paired 语义字段：
  ```python
  identity_col
  timepoint_col
  pre_condition
  post_condition
  ```

**5. Metadata cache 持久化层**
[datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:198) 的 `_save_to_cache`  
[datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:227) 的 `_load_from_cache`

已完成：

- cache 保存/读取新增字段：
  ```python
  patient_ids
  timepoints
  paired_group_condition_identity_pool
  ```
- cache config 校验从“只比较 dataset_configs 数量”升级为比较完整 `DatasetConfig` 字段，避免同路径但 paired 参数不同还误用旧 cache。

**6. Paired 查询索引层**
[datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:491) 的 `_precompute_acceleration_pools`

已完成：

- 新增 paired 专用索引池：
  ```python
  paired_group_condition_identity_pool
  ```
- key 是：
  ```python
  (group_id, condition/timepoint, identity/cell_type)
  ```
- 也就是可以快速查：
  ```text
  某个 Patient + Pre-biopsy + T cell 的所有 cell indices
  某个 Patient + Post-surgery + T cell 的所有 cell indices
  ```

新增访问函数：

- [get_patient_ids](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:1060)
- [get_timepoints](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:1064)
- [get_paired_condition_identity_cells](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:1068)

**7. Replacement 逻辑层**
[datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:833) 的 `find_replacement_cells`  
[datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:1003) 的 `_get_candidate_pool`

已完成：

- 原来只支持：
  ```text
  human: different donor replacement
  drug: control condition replacement
  ```
- 现在增加：
  ```text
  paired: same Patient + same cell_type + Pre-biopsy replacement
  ```
- 也就是说，当前 paired 语义是：
  ```text
  ground truth sample 来自 Post-surgery
  observed/replacement cell 来自同 Patient 的 Pre-biopsy
  cell_type 必须匹配
  ```

**8. Split 和 sample 生成层**
[datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:1244) 的 `_split_groups`

已完成：

- 新增 `paired_groups`。
- paired 数据按 Patient group 切分，避免同一个 Patient 的 pre/post 跨 train/val/test 泄漏。

[datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:1309) 的 `_generate_samples_by_locality`

已完成：

- paired 模式下，只用 `Post-surgery` 细胞生成 sample。
- `Pre-biopsy` 不再进入 ground truth sample，只作为 replacement/input 候选池。
- 日志统计增加 paired sample 数。

**9. Singleton cache key**
[datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:1182)

已完成：

- cache key 从只看 `path + target_genes`，改成也包含完整 `DatasetConfig` 语义。
- 避免同一路径从 `human` 改成 `paired` 后在同一进程中误复用旧 cache。


已按“最小改动”实现 **paired 数据的 7-fold LOPO split**。现在一次运行对应一个 fold，`fold_index=0..6` 轮流把 1 个 Patient 作为 test，`val_fold_offset=1` 取下一个 Patient 做 val，其余 5 个 Patient 做 train。

改动位置：

- [launch_finetuning.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/cli/launch_finetuning.py:134)  
  新增 CLI 参数：`--split_strategy {random,lopo}`、`--fold_index`、`--val_fold_offset`，并传入 DataModule。也补了 paired 配置日志。

- [datamodule.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/finetune/datamodule.py:28)  
  接收并保存 `split_strategy/fold_index/val_fold_offset`，在 `setup()` 里传给 dataset 构造函数。

- [datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:1317)  
  在 `_split_groups()` 里新增 `split_lopo_groups()`：按 Patient `original_id` 稳定排序；`test = fold_index`；`val = fold_index + val_fold_offset`；`train = 剩余患者`。如果 paired 患者少于 3 个或 fold 越界，会直接报错。

- [datasets.py](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/src/stack/data/finetuning/datasets.py:2227)  
  `create_train_val_test_datasets()` 也接收并传递 LOPO 参数，保证 train/val/test 三个 Dataset 共享同一组 patient split。

- [ft_parsecg.yaml](/Users/zhangyaojie/Desktop/virtual_cell/Lung_stack/configs/finetuning/ft_parsecg.yaml:15)  
  默认启用：
  ```yaml
  split_strategy: lopo
  fold_index: 0
  val_fold_offset: 1
  ```


datasets.py (line 34)新增 PAIRED_SAMPLING_METHODS 四种可选方法。

在 MultiDatasetSplittableDataset 增加 paired_sampling_method 参数与校验。


新增 _build_paired_balanced_sample() 及四个清晰标注的方法：

method1_capacity_strict

method2_capacity_repeat_pre

method3_post_only_strict

method4_post_only_repeat_pre

_get_common_paired_pools

_allocate_counts_by_capacity

_assemble_paired_balanced_sample

_build_paired_balanced_sample


paired 数据在 __getitem__() 中改走新分支，绕开原来的 find_replacement_cells() 和 replacement 后 v8 重排，保持：left = [post prompt | post target]
right = [same post prompt | matched pre query]
前 n_kept 始终是 prompt，后面始终是 query。


datamodule.py (line 31)新增并向下传递 paired_sampling_method。

launch_finetuning.py (line 149)新增 CLI 参数 --paired_sampling_method，并在日志里打印当前策略。

ft_parsecg.yaml (line 18)当前默认设为 paired_sampling_method: method1_capacity_strict。

新增了test_paired_sampling_methods.py (line 122)增加四个采样策略测试，覆盖 prompt/query 边界、cell type 对齐、strict 不重复 pre、repeat 方法允许 pre 重复补齐。




datasets.py (line 1248)：在 dataset 初始化时加入 paired preflight validator，提前检查 LOPO train/val/test 是否有 paired group、每个 patient 的 common pre/post cell-type pool 是否能满足 sample_size / n_kept / n_replaced，并能提前暴露类似 P09 only 393 post cells but sample_size=512 的问题。
datasets.py (line 1806)：把 paired 四种采样方法的 allocation 逻辑抽成共享 helper，preflight 和真正采样共用同一套判定，避免验证逻辑和运行逻辑漂移。


datasets.py (line 2975)：给 create_datasets_from_gene_list() 补了 split_strategy / fold_index / val_fold_offset 参数，并向下传给 create_train_val_test_datasets()。


launch_finetuning.py (line 277)：新增 checkpoint n_genes 与 data_module.n_genes 的一致性校验。


launch_finetuning.py (line 374)：在加载 checkpoint config 后、真正 load model 前调用维度校验，不一致会直接报错。

测试也补上了：

test_paired_sampling_methods.py (line 315)：paired config parse。

test_paired_sampling_methods.py (line 352)：LOPO paired split 非空检查。

test_paired_sampling_methods.py (line 379)：sample_size=512 下 P09 post cell 不足的预期报错。

同文件还补了 strict method capacity failure、pass-through 参数转发、checkpoint gene dimension mismatch 的测试。




in-context prediction/generation 默认输出 NB mean，而不是 sampled count；同时保留 sample 选项。

改动位置：


inference.py (line 552)：get_incontext_prediction(..., prediction_output="mean")

inference.py (line 695)：默认 result = mean_preds[is_test_cell_mask]，sample 时才用 count_preds

inference.py (line 848)：get_incontext_generation() 继续向下传递 prediction_output

generation.py (line 291)：CLI generation API 增加 prediction_output

generation.py (line 462)：新增命令行参数 --prediction-output {mean,sample}



已按 Stage1 保守微调策略完成修改：默认只训练 query_pos_embedding、cls、output_mlp 最后一层，冻结 Stack 主干和 decoder 前半部分。

修改位置

src/stack/finetune/lightning.py (line 23)
新增 head_lr、decoder_lr、finetune_strategy 参数。
src/stack/finetune/lightning.py (line 60)
新增 _apply_finetune_strategy()：
stage1 下先冻结 self.model.parameters()，再打开：
query_pos_embedding、cls.parameters()、output_mlp[3].parameters()。
src/stack/finetune/lightning.py (line 85)
新增 _log_trainable_parameter_count()，启动时会打印可训练参数比例。
src/stack/finetune/lightning.py (line 263)
修改 configure_optimizers()：Stage1 使用 3 个 param groups：
query_pos_embedding 用 head_lr 且 weight_decay=0；
cls 用 head_lr；
output_mlp[3] 用 decoder_lr。
src/stack/cli/launch_finetuning.py (line 191)
新增 CLI 参数：--finetune_strategy {stage1,full}、--head_lr、--decoder_lr。
src/stack/cli/launch_finetuning.py (line 423)
checkpoint 加载和 scratch 初始化都传入上述新参数。
configs/finetuning/ft_parsecg.yaml (line 12)
显式设为：
finetune_strategy: stage1、head_lr: 0.0001、decoder_lr: 0.00001。
tests/test_finetune_stage1.py (line 1)
新增轻量单元测试，只检查冻结策略和 optimizer 分组，


之前 PyTorch 默认 collate 会尝试把 batch 内所有 metadata dict 深度合并，要求每个样本都有相同 keys，于是某个样本没有 Neutrophil 就崩。
在 datamodule.py (line 16) 加了 finetune_collate_fn：tensor 正常 torch.stack，metadata 保留为 list。train/val/test dataloader 都改成使用这个 collate。


根因在 lightning.py (line 151)：保存 checkpoint 时只保留 model.*，但 trainer.test(... ckpt_path="best") 会严格加载完整 LightningModule，需要 teacher_model.*。
在 lightning.py (line 156) 加了 on_load_checkpoint()：如果 checkpoint 只有 model.*，加载时自动复制一份到 teacher_model.*。这样仍然保持 checkpoint 小，但 test 可以正常加载。




新增了独立采样模块 sample_id_sampling.py (line 92)：GroupedTestSamplerDataset 会按 obs["sample_id"] 分组生成 cell set，最后不足 n_cells 的组内尾巴仍会补采样给模型，但保存 embedding 时只保留真实 unique cells，所以不会让 Pre/Post 或不同 sample_id 互相作为 Stack context。
我也接入了 embedding CLI：embedding.py (line 262) 新增：
bash

--groupby-obs-col sample_id
--patient-col Patient

使用方式示例：
bash

PYTHONPATH=src /data/home/zhangyaojie/miniconda3/envs/Stella/bin/python -m stack.cli.embedding \
  --groupby-obs-col sample_id \
  --patient-col Patient \
  --output /data/home/zhangyaojie/Lung_stack/outputs/embedding/embedding/NG.h5ad

输出会按这个结构保存：
text

outputs/embedding/embedding/NG/<Patient>/<sample_id>.h5ad

例如 NG/P01/BD_P_P01_N.h5ad。

新增了 tests/test_sample_id_sampling.py (line 13)，但当前 base 和 Stella 环境都没有 pytest，所以 pytest 测试文件本身未能用 pytest runner 执行。



在 src/stack/classifier/ 新增独立分类模块，不改现有 Stack 模型主体：

dataset.py

从已经筛选为 Pre-biopsy 的 .h5ad 读取数据。
按 sample_id 内部连续切成 cell set，避免不同样本互相作为 context。
每个 cell set 的 label 来自该 sample_id 对应的 adata.obs["response"]。
只接受 NMPR、MPR，映射为 0/1。
sample_id in {"XGY_P_P05_P", "BD_immune08"} 固定为 test，其余为 train。
不设置 validation，不做 CV。
model.py

包装原始 Stack encoder。
输入 cell set counts，经过 Stack encoder 得到 (batch, n_cells, 1600)。
对 n_cells 维度 mean pooling，得到 (batch, 1600)。
接 nn.Linear(1600, 2) 得到分类 logits。
Stack encoder 和分类头一起训练，全量微调。
train.py

CLI 脚本：加载 checkpoint、构建 train/test dataloader、训练全模型。
loss 用 CrossEntropyLoss。
输出 test accuracy、每个 sample_id 的预测结果、保存 checkpoint。
不引入 Lightning，先用原生 PyTorch，代码更直接，和现有模型解耦。
我会避免的事情

不改 src/stack/models/core 的 Stack 架构，不改 finetune 现有 paired 逻辑，不把这个分类任务混进生成/finetune pipeline。

新增代码都放在 src/stack/classifier/，没有改 Stack 原始模型架构。
新增内容：
dataset.py (line 41)：按 sample_id 生成 cell set，标签来自 obs["response"]，默认 NMPR=0, MPR=1；固定 XGY_P_P05_P 和 BD_immune08 为 test，其余为 train；不做 Timepoint 过滤。
model.py (line 10)：包装 Stack encoder，输出 (batch, n_cells, 1600) 后对 cell 维度 mean pooling，再接 Linear(1600, 2)。
train.py (line 172)：原生 PyTorch 训练脚本，Stack encoder + 分类头全部参与训练，不冻结；无 CV、无 validation。
tests/test_classifier_dataset.py (line 23) 和 tests/test_classifier_model.py (line 17)：覆盖固定 test split 和 mean pooling 行为。
运行示例：
bash



PYTHONPATH=src /data/home/zhangyaojie/miniconda3/envs/Stella/bin/python -m stack.classifier.train \
  --checkpoint /path/to/bc_large.ckpt \
  --adata /path/to/pre_biopsy_only.h5ad \
  --genelist /path/to/basecount_1000per_15000max.pkl \
  --gene-name-col gene_symbols \
  --output-dir /data/home/zhangyaojie/Lung_stack/outputs/classifier/pre_response \
  --epochs 10 \
  --batch-size 8 \
  --lr 1e-5 \
  --device auto

输出：
stack_cellset_classifier.pt
metrics.json
test_predictions.csv
验证已完成：
unittest tests.test_classifier_dataset tests.test_classifier_model：2 tests OK
py_compile：通过
python -m stack.classifier.train --help：通过



已经完成的部分
dataset.py 已完成固定 train/test 拆分逻辑：CellSetClassificationDataset 会按 sample_id 生成 cell set，默认把 XGY_P_P05_P 和 BD_immune08 放进 test，其余 sample_id 放进 train；不做 CV、不做 validation。标签来自 obs["response"]，默认映射是 NMPR=0, MPR=1。它复用了之前的 gene 对齐和 h5ad 读取逻辑，所以输入还是原始表达矩阵 cell set，不是预先保存的 per-cell embedding。

model.py 已完成你的核心模型想法：StackCellSetClassifier 包装原始 Stack encoder，输入 cell set 后走 _reduce_and_tokenize() 和 _run_attention_layers()，得到 (batch, n_cells, embedding_dim) 的 cell embeddings，然后对 n_cells 维度 mean pooling，最后接 Linear(embedding_dim, 2)。embedding_dim 自动用 stack_model.n_hidden * stack_model.token_dim，你的大模型应当就是 1600。

train.py 已完成一个基础训练脚本：加载 Stack checkpoint，构建 train/test dataloader，用 CrossEntropyLoss 和 AdamW 训练。因为 optimizer 是 model.parameters()，所以 Stack encoder 和线性分类头都会被训练，也就是符合你后来要求的“全量微调，不冻结 Stack encoder”。训练后会保存 stack_cellset_classifier.pt、metrics.json 和 test_predictions.csv。


26.7.8
完成了修改：
dataset.py：支持 train/val/test 显式 sample split，新增 read_sample_label_map() 和固定 test + 7-fold val split helper。
model.py：加入 6 种分类头，默认冻结 Stack encoder，只训练 classifier head。
train.py：升级为固定 test、7-fold val、early stopping、超参 grid、TensorBoard、patient-level metrics/predictions 输出。
__init__.py：导出新增 helper/head 类型。
关键一点：preneo_IAT.h5ad 实际类别数是 MPR=10, NMPR=8, pCR=9。每类固定 2 个 test 后，NMPR 只剩 6 个，所以严格“不重复的 7-fold 且每 fold val 每类 1 个”数学上不可能。我实现为 7 个 stratified validation folds，不足类会循环复用，split_summary.json 会记录 repeated/never-val samples。


1. 一共27例患者样本，label为3类：nmpr/mpr/pcr，数据划分为train:val:test = 18:3:6, val中3个labels每种一例，test中每种label两例；固定3类labels中每类 2 例作为 test，然后剩余21例做7-fold CV，每一fold先选出3例val，要求每种标签一例，然后剩余的作为train。
2. 分类头设置以下几种不同的分类头：
   1. 简单linear线性层；
   2. Linear + activation + dropout + Linear；
   3. self.cls = nn.Sequential(nn.Linear(hidden_dim, middle_dim),
        nn.LayerNorm(middle_dim),
        nn.GELU(),
        nn.Dropout(),
        nn.Linear(middle_dim, num_classes)
     )
    4. self.cls = nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, num_classes)
     )
    5. self.cls = nn.Sequential( nn.Dropout(), nn.Linear(hidden_dim, num_classes) )
    6. Attention pooling + classification head: class AttentionPoolingHead(nn.Module): def __init__(self, hidden_dim, num_classes): super().__init__() self.attn = nn.Sequential( nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1) ) self.cls = nn.Linear(hidden_dim, num_classes) def forward(self, cell_emb): # cell_emb: [n_cells, hidden_dim] attn_score = self.attn(cell_emb) # [n_cells, 1] attn_weight = attn_score.softmax(dim=0) # [n_cells, 1] sample_emb = (attn_weight * cell_emb).sum(dim=0) logits = self.cls(sample_emb) return logits, attn_weight
3. 使用stack模型已经预训练好的encoder参数，冻结encoder，仅仅训练classifier分类头部分；
4. 需要 val 来做：early stopping、选择 epoch、选择学习率、weight decay、选择 sample_size / max_samples；选择 cell set 数量；选择分类头结构
5. 最后结果做patient-level accuracy(对每个患者的所有 cell sets 取平均预测概率,然后得到 patient-level prediction)；macro-F1；balanced accuracy；per-class recall；confusion matrix；
6. 调用tensorboard实时监控学习中的各项参数

验证已跑：
compileall src/stack/classifier: OK
tests/test_classifier_dataset.py: OK
tests/test_classifier_model.py: OK
python -m stack.classifier.train --help: OK
真实 IAT split check: OK
auto test ids with seed 0 = ('P41', 'P52', 'P4', 'P35', 'P43', 'P46')
