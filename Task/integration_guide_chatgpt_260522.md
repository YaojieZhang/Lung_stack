# 小样本场景下基于 Stack 与 scPhase 的最小改动建模分析

## Executive summary

对你现在这组数据，真正的“小样本”并不在细胞层面，而在**patient-level 监督只有 7 个病人**。因此，最优策略不是端到端重训一个新大模型，而是把 **Stack 保留为主干的条件生成器**，利用其已有的 ICL / frozen-teacher fine-tuning 机制去学习“治疗前 → 治疗后”的单细胞状态迁移；再把 **scPhase 保留为 patient-level MIL 聚合器**，专门负责从 cell bag 预测 MPR / NMPR。这样既复用 Stack 的大规模预训练归纳偏置，也把少量 patient 标签集中给更小、更可控的 patient classifier。这个判断与 Stack 官方 README、公开 fine-tune 代码，以及 scPhase 论文与仓库中对 patient-as-bag、MoE-MIL、domain adaptation 的设计是一致的。citeturn5view0turn5view2turn4view1 fileciteturn21file0L3-L3 fileciteturn24file0L3-L3 fileciteturn16file0L3-L3

在“尽量不改原架构”的前提下，我认为最值得做的只有三件事。第一，**Stack 只做轻量微调**，优先更新 `query_pos_embedding`、输出解码头、fine-tune 分类头，以及最后 1–2 个注意力块的 LoRA / Adapter；不要一开始全量解冻。第二，**scPhase 不要照搬 8 个专家**，在 7 个病人的任务上更合理的是 **2–4 个专家**，否则极易 expert collapse。第三，**下游不要直接吃 Stack 随机采样出来的 counts**，而要用 Stack 预测分布的**均值**，并把 `治疗前表达 + 预测后表达 + delta` 作为 scPhase 的 cell-level 输入，而不是只用预测后的表达。fileciteturn25file0L3-L3 fileciteturn35file0L3-L3 fileciteturn36file0L3-L3 citeturn7academia0turn8academia1

## 问题本质与原架构约束

Stack 的公开定位是一个**训练于 1.5 亿单细胞**的大规模 encoder-decoder foundation model，核心是 tabular attention：在 `TabularAttentionLayer` 中交替进行 cell-wise 与 gene-wise attention；其 fine-tuning 路径不是普通 supervised head，而是**学生-教师双模型**，教师冻结，训练开始时从学生同步，并周期性做 EMA 更新。代码里，fine-tune 总损失写得非常清楚：`recon_loss + mmd_loss + sw_weight * sw_loss + cls_loss`。这意味着 Stack 已经内置了一套相当强的“条件生成 + 表示对齐”机制，不应该在你这个 7-patient 项目里被轻易推翻。citeturn5view0turn5view1turn5view2 fileciteturn27file0L3-L3 fileciteturn29file0L3-L3 fileciteturn24file0L3-L3 fileciteturn21file0L3-L3

scPhase 则是一个非常标准、也非常适合你当前问题的**patient-as-bag** 框架：单细胞表达先过 instance encoder，再经 LinFormer attention，再由 MoE-MIL 聚合，最后做 patient-level 分类，并可选域对抗校正 batch effect。论文明确强调其目标是从单细胞 bag 直接预测 clinical phenotype，而 README 和配置文件则显示默认输入是 5000 HVGs、`hidden_dim=256`、`num_heads=8`、`linformer_k=128`、`moe_num_experts=8`、`lr=1e-5`。也就是说，scPhase 的架构主体并不需要你重写，它的问题只是在你这个超小 patient 数设定下**容量偏大**。citeturn4view1turn5view3turn18view0 fileciteturn16file0L3-L3 fileciteturn36file0L3-L3

真正需要你注意的约束反而有两个。其一，**Stack checkpoint 的 gene 空间与 scPhase 默认 5000 HVGs 不天然一致**。Stack 的训练/微调都依赖固定的 `genelist_path`，而 scPhase 更像是一个可改 `input_dim` 的 MLP-MIL 模型。因此更稳的做法是：**Stack 生成时保留 checkpoint 原始基因空间；scPhase 再在训练折内从与 Stack 重叠的基因里选 2k–5k 个基因子集**。其二，scPhase 官方 `run_cv.py` 的自动 split 逻辑是围绕 `batch_col` / group 做的，而你的真实无泄漏单位应当是 **patient**。这意味着你不能直接把官方 CV 脚本当外层评估器，而要自己包一个 patient-level 外层切分。fileciteturn17file0L3-L3 fileciteturn18file0L3-L3 fileciteturn38file0L3-L3 fileciteturn39file0L3-L3

## 推荐的新架构

在“最小改动”原则下，我推荐的新架构可以概括为：**Stack 负责生成治疗后细胞状态，scPhase 负责聚合病人反应，二者之间只加一个轻量 Bridge，而不改写主干。**

| 模块 | 原始设计 | 推荐改法 | 为什么这是最优折中 |
|---|---|---|---|
| Stack 主干 | 保持原始 tabular attention 与 frozen-teacher fine-tune | **不改** | 预训练价值最大，7 病人不足以支撑重训 |
| Stack 输出 | 默认 ICL generation 最终常用 sampled counts | 改为**输出 NB mean**，并保留方差摘要 | 降低下游 patient-level 随机噪声 |
| Bridge | 原始无此层 | 将 `pre`、`pred-post`、`delta` 拼接后送入 scPhase | 疗效本质是 treatment-induced change，而非静态 post |
| scPhase MoE-MIL | experts 默认 8 | 改为 **2–4 experts** | 7 病人监督不足，8 专家过大 |
| scPhase domain adaptation | 论文/代码支持 GRL | 仅在训练折内确有多 batch 时开启 | 防止过度校正掉真实治疗效应 |

最关键的新增，不是新 backbone，而是 **Bridge 特征定义**。我建议每个细胞的下游特征用

\[
f_c = [x^{pre}_{c,S},\ \hat{x}^{post}_{c,S},\ \Delta_{c,S}],
\qquad
\Delta_c = \hat{x}^{post}_c - x^{pre}_c,
\]

其中 \(S\) 是训练折内筛出来的基因子集。也就是说，scPhase 不只看预测后的细胞，而是显式看“治疗前背景”和“治疗变化量”。这是一个非常小的接口修改，却通常比“只喂预测后表达”更贴近疗效机制，因为 MPR / NMPR 本质上是对**治疗诱导变化**的表征，而不仅是绝对状态。scPhase 的 instance encoder 本来就是一个 MLP，`input_dim` 可配，所以这种多通道输入对原架构几乎没有破坏。fileciteturn16file0L3-L3 fileciteturn36file0L3-L3

如果你想再进一步、但仍然不大改模型，我认为最值得做的增强是**双 prompt bank 推理**。也就是在 Stack ICL 生成时，不只用“所有训练病人的 post-treatment 细胞”做 prompt，而是分别用 **MPR-prompt bank** 和 **NMPR-prompt bank** 生成两套候选后治疗表型，再把两套候选及其差值交给同一个 scPhase 头。这个思路仍然没有改 Stack 或 scPhase 的结构，只是改变了 prompt 的组织方式，而且很贴合你的任务语义：不同反应类别可能对应不同的治疗后细胞状态吸引子。citeturn5view0turn4view1

## 微调策略与关键超参数

### 最推荐的冻结与解冻顺序

对 7-patient 数据，最合理的微调顺序不是“从头到尾全开”，而是分为三个层级。第一层级，**全冻结 Stack，只训练 scPhase**，这一步只是做 pipeline sanity check，看看生成结果是否至少为 patient-level 预测提供了可学信息。第二层级，也是我最推荐的默认层级，解冻 Stack 中最任务相关、但参数量最小的部分：`query_pos_embedding`、`output_mlp`、fine-tune `cls`，再在最后 1–2 个注意力块插入 LoRA 或 Adapter。第三层级，仅当内层验证集上连续稳定提升时，再额外解冻最后 1 个完整 Stack block。LoRA 和 Adapter 的共同优点是：它们都让你**保留冻结主干**，只用很少的新增参数吸收新任务偏差，这和 Stack 代码中“教师冻结、学生轻调”的方向高度一致。fileciteturn21file0L3-L3 fileciteturn25file0L3-L3 citeturn7academia0turn8academia1

我建议的参数级别如下。

| 组件 | 推荐设置 | 搜索范围 |
|---|---|---|
| Stack `sample_size` | 256 | 固定 256 |
| Stack `replacement_ratio` | **0.625** | 0.50 / 0.625 / 0.75 |
| Stack LoRA rank | 4 | 4 / 8 |
| Stack 训练层 | `query_pos_embedding` + `output_mlp` + `cls` + 最后2层LoRA | 逐步加一层完整 block |
| Stack LR | `1e-4`（head），`5e-5`（LoRA），`1e-5`（完整 block） | 同量级 |
| Stack weight decay | `1e-3` 到 `3e-3` | `1e-4` 到 `3e-3` |
| Stack epochs | 8–15 | 不超过 20 |
| scPhase experts | **2–4** | 1 / 2 / 4 / 8 |
| scPhase hidden dim | 256 | 固定 256 |
| scPhase `linformer_k` | 64 | 64 / 128 |
| scPhase LR | `1e-5` 到 `3e-5` | 同量级 |
| scPhase weight decay | `1e-4` | `1e-5` 到 `3e-4` |
| scPhase epochs | 30–60 | 不超过 80 |
| Batch size | 1（scPhase），8（Stack） | 尽量不涨 |

这里的依据很直接：Stack 官方配置示例使用 `sample_size=256` 预训练、`sample_size=512` 微调，scPhase 公布的 NSCLC 配置则是 `input_dim=5000`、`hidden_dim=256`、`moe_num_experts=8`、`lr=1e-5`。在你这个 7-patient 任务里，保留这些量级但**下调 MIL 容量**，通常是更稳的。fileciteturn39file0L3-L3 fileciteturn38file0L3-L3 fileciteturn36file0L3-L3

### 学习率调度与正则化

学习率调度方面，我建议保持两边都用 **AdamW + cosine**。scPhase 论文和代码本来就是 AdamW + cosine + warmup，Stack fine-tune CLI 也支持 cosine / ReduceLROnPlateau，而且公开配置用了 cosine。citeturn4view1 fileciteturn17file0L3-L3 fileciteturn31file0L3-L3 fileciteturn38file0L3-L3

具体到正则化，我的推荐很明确：

- Stack 端优先保留其自带的 gene masking、MMD、SW；
- scPhase 端保留 `instance_dropout_rate`，但把分类头 dropout 维持在 `0.1–0.2`，不要上到 `0.3+`；
- 所有梯度裁剪统一 `clip_grad_norm = 1.0`；
- 早停要比公开大数据配置更激进：Stack `patience=3–5`，scPhase `patience=8–12`；
- 若类别不平衡明显，用 **Class-Balanced Loss** 或 class-weighted CE，而不是只靠 oversampling。citeturn9academia0

若你希望把 Stage A 和 Stage B 做成一个弱联合目标，我建议不要手工拍脑袋定损失权重，而用**基于不确定性的多任务加权**：

\[
L = e^{-s_1}L_{\text{stack}} + s_1 + e^{-s_2}L_{\text{patient}} + s_2,
\]

其中 \(s_1, s_2\) 为可学习标量。这种做法在多任务学习里是一个很成熟的稳定方案。citeturn9academia1

### 少样本场景里哪些增强值得做，哪些不值得

在少样本项目里，增强策略并不是越多越好。对你这里，我建议优先级如下：

第一优先级是 **PEFT**，也就是 LoRA / Adapter。这个已经在上面说过，是最值得做的。citeturn7academia0turn8academia1

第二优先级是**重采样而不是重合成**。你自己的数据量在细胞层面并不小，所以更好的做法是通过 `resample_each_epoch`、patient-stratified cell set sampling 和同类 bag mixup 去增加训练视角，而不是立刻上 GAN。Stack datamodule / lightning 里本就支持 epoch 级重采样，这和你的 “~200 cell sets” 设定非常匹配。fileciteturn40file0L3-L3 fileciteturn21file0L3-L3

第三优先级是**mixup 只作为 ablation**。mixup 本身是靠谱的正则化方法，但在 7 个病人的 patient-level 任务里，我只建议做**同类 patient 或同类 pseudobulk** 的 mixup，别做跨类 mixup。citeturn8academia0

第四优先级才是**合成数据**。如果你一定要做，我也更倾向于把它当作少量稀有 cell type 的补充，而不是主训练数据来源。更现实的替代是直接用条件生成的单细胞模型，如 scDiffusion 一类方法；但无论用什么，必须严格限制在训练折内拟合，不能把测试病人的分布信息带回来。citeturn16academia3

MAML 这类元学习我不建议作为主方案。它适合“有很多 task、每个 task 都数据很少”的场景，而你只有 7 个病人，能形成的 meta-distribution 太窄，更像是高方差试验而不是稳定收益来源。citeturn8academia2

## 训练新模型时最需要警惕的问题

最需要警惕的第一个问题是**patient leakage**。只要你的外层评估不是严格 leave-one-patient-out，或任何一项 preprocessing 是在全体 7 个病人上先做完再切分，那么最终得到的 patient-level 指标都会偏乐观。尤其 scPhase 官方脚本倾向按 batch / group 来切，而不是按 patient 切，这一点必须你自己接管。fileciteturn17file0L3-L3

第二个问题是**把 batch correction 做得过强**。scPhase 论文本身就把“传统 batch correction + clustering 可能引入解释偏差”作为其问题背景之一，并为此加了 domain adaptation。对你这个 pre/post 治疗项目，真正的治疗效应和技术批次往往纠缠在一起。因此我建议：**Stack 端保留原始 counts 语义，不做强表达空间校正；scPhase 端只在确有多 batch 时启用 GRL；Harmony / scVI 等方法只用于聚类、pseudo-type 或可视化，不直接替代训练输入矩阵。**citeturn4view1

第三个问题是**把 attention 当成最终解释**。你虽然这条消息没有专门问解释性，但如果之后会回溯“哪些细胞/基因决定疗效”，那现在就应该定下基调：attention 可以作为很好的 instance importance 线索，但不能单独代表因果式解释；更稳妥的是 attention + Integrated Gradients + leave-one-cell-out 反事实。IG 的优点在于不需要改模型，而 attention-only 的局限在解释性研究中早就被指出。citeturn6academia0turn20academia0

第四个问题是**Stack generation 的随机采样噪声**。这点我再强调一次：默认 sampled count 对可视化未必有问题，但对 7-patient 的 patient-level classifier 很不友好。你后续若要训练一个新模型，建议把 Stage B 的标准输入固定为 **NB 均值输出**，把采样方差当作附加 uncertainty feature，而不是把单次采样计数直接喂给 scPhase。fileciteturn35file0L3-L3

第五个问题是**专家数与输入维度一起膨胀**。你如果把 `pre + pred + delta` 全部喂进去，同时还让 scPhase 保持 `5000 × 3` 输入和 `8 experts`，那很可能 validation patient 一换就崩。我的建议是先把下游基因数压到 2k，再把 experts 压到 2–4；如果这都稳定了，再扩到 5k 或 8 experts。对 7 病人项目，容量一定要一步一步加，而不是一步到位。fileciteturn36file0L3-L3

## 结论

如果你现在只问“先分析问题”，我的结论可以压缩成一句话：**最佳的新模型不是全新模型，而是“Stack 轻量条件生成 + scPhase 缩容 MIL 聚合 + 多通道 bridge”的最小改动组合。**Stack 保持主架构不动，只做 PEFT 级微调；scPhase 保持 patient-level MIL 不动，只把 experts 从 8 降到 2–4，并把输入从 `pred-post only` 改成 `pre + pred-post + delta`；整个系统用 patient-level 外层 CV 控制泄漏，用 class-balanced loss、cosine warmup、激进早停和弱联合不确定性加权控制过拟合。fileciteturn24file0L3-L3 fileciteturn25file0L3-L3 fileciteturn16file0L3-L3 fileciteturn36file0L3-L3 citeturn7academia0turn8academia1turn9academia0turn9academia1

若你接下来要进入下一步，我建议优先落地四个动作：先固定 **patient-level split**；再确定 **Stack checkpoint 与基因空间**；然后实现 **NB mean generation** 的小补丁；最后搭建一个 **scPhase 2-expert、2k genes、pre+pred+delta 输入** 的最小可跑版本。只要这四步跑通，你后面的所有消融都会有清晰锚点。