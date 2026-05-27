# Stack: In-Context Learning of Single-Cell Biology

Mingze Dong1,2, Abhinav Adduri1, Dhruv Gautam1,3, Christopher Carpenter1, Rohan Shah1,4, Chiara Ricci-Tam1, Yuval Kluger2, Dave P. Burke‡,1, Yusuf H. Roohani‡,†,1 

1Arc Institute; 2Yale University; 3University of California, Berkeley; 4University of Pennsylvania 

Single-cell transcriptomics offers the promise of measuring the diversity of cellular phenotypes across species, diseases, and other biological conditions. Recently, foundation models have emerged to identify this variation, yet most methods represent each cell independently, despite technical limitations that reduce measurement precision at the single-cell level. Here, we present Stack, a foundation model trained on 149 million uniformly preprocessed human single cells that leverages tabular attention to generate representations for each cell informed by the cells in its context. Stack offers substantial improvements for downstream tasks in the zero-shot setting compared to baselines, whether they are zero-shot, fine-tuned, or trained from scratch on the target dataset. Stack can perform in-context learning from unlabeled cells representing arbitrary conditions, such as a chemical perturbation or a different donor, and predict the effect of those conditions on a target cell population without requiring data-specific fine-tuning. We apply Stack to generate Perturb Sapiens, the first human whole-organism atlas of perturbed cells, spanning 28 tissues, 40 cell classes, and 201 perturbations. We validated subsets of Perturb Sapiens using in vitro stimulation profiles. Overall, Stack presents a new modeling framework where cells themselves act as guiding examples at inference time, unlocking general-purpose in-context learning capabilities for single-cell biology. 

# 1. Introduction

The advent of large-scale single-cell RNA sequencing has generated unprecedented volumes of cellular data, with collections now exceeding hundreds of millions of cells across diverse tissues and conditions (Program et al., 2025; Youngblut et al., 2025; Consortium* et al., 2022). This data explosion has enabled the development of single-cell foundation models that leverage self-supervised learning to extract meaningful cellular representations from massive datasets (Theodoris et al., 2023; Cui et al., 2024; Rosen et al., 2023; Hao et al., 2024; Ding et al., 2025). These models are subsequently fine-tuned for various downstream tasks such as cell type annotation, batch integration, and perturbation effect prediction. 

Despite their promise, current single-cell foundation models face significant limitations in their capacity for biological discovery. They often fail to surpass classical approaches when employed in a zero-shot manner (Kedzierska et al., 2025; Liu et al., 2023). Even with dataset-specific fine-tuning, they struggle to improve over simple baselines in perturbation prediction (Wu et al., 2024; Li et al., 2024; Kernfeld et al., 2023; Ahlmann-Eltze et al., 2025). Recent models (such as Adduri et al., 2025 and Klein et al., 2025) show improvements in these capabilities; however, they still require extensive training data and supervision on biological conditions and tasks of interest. This limits their potential for de novo biological discoveries, such as inferring perturbation effects in novel biological conditions (e.g. unseen cell types for which only observational data is available) or performing novel tasks, such as predicting sample-specific variation in immune phenotypes. 

Most current single-cell foundation models are constrained by fundamental design choices. First, their pre-training objectives operate at the single-cell level, training models to function as universal “denoisers” that exploit gene dependencies but cannot see shifts at the population scale. Second, the inherently noisy count distribution of gene expression profiles necessitates aggregating information across cells to enhance signalto-noise ratios of gene expression patterns. This insight has been employed in State for perturbation effect prediction (Adduri et al., 2025), but remains underexplored for single-cell foundation models. This aggregation is also essential for encoding inter-cellular interactions or mutual information that would otherwise be neglected or misattributed to gene-level dependencies. Third, prior models have largely been used for downstream tasks through task-specific fine-tuning on limited data, rather than exploiting the demonstrated ability of large transformer based models to perform robust learning at inference time. 

To address these limitations and progress towards cellular foundation models that generalize beyond their supervised training conditions and tasks, we developed Stack, a self-supervised framework that enables in-context learning through a novel architecture on cell sets. Pre-trained on scBaseCount (Youngblut et al., 2025), the largest existing single-cell collection with 189 million high-quality human cells, Stack introduces several key innovations. The architecture employs customized transformer blocks that account for both intercellular and intra-cellular information flows, inspired by recent advances in tabular deep learning (Hollmann et al., 2025; Qu et al., 2025). A novel pre-training objective prevents simple memorization shortcuts while maintaining single-cell resolution. Additional improvements include enforcing linear identifiability in the latent space for better generalization, and a highly efficient dataloader for scalability. 

After pre-training, Stack demonstrates the ability to automatically leverage cellular context information at the time of inference to refine embeddings and enable several downstream tasks, even for datasets never encountered during training. Through extensive evaluations, we show that this property enables substantial performance improvements in zero-shot cellular classification and integration tasks compared to various baseline models, whether they are zero-shot, fine-tuned, or even trained from scratch on the evaluation dataset. 

Besides leveraging the context of a cell to enhance its embedding, we can also engineer the cell’s context to influence its state. Through a novel post-training alignment procedure, Stack introduces in-context learning (ICL) for single-cell foundation models. By post-training on an annotated 55-million-cell collection from CellxGene and the Parse peripheral blood mononuclear cell (PBMC) perturbation dataset (Program et al., 2025; Parse Biosciences, 2023), Stack learns to function as a conditional generative model analogous to masked diffusion models (Sahoo et al., 2024) at the cell dimension. The alignment enables “cell prompting” tasks, where Stack takes two cell populations termed prompt and query, and predicts how the query population would behave under the condition represented by the prompt. The framework empowers general tasks such as perturbation effect prediction and condition-specific cellular profile generation, supporting both perturbed and observational cells as prompts. 

Our evaluation reveals that Stack can generate unseen context- or perturbation-specific cell types after post-training, on completely unseen data without data-specific fine-tuning. Across perturbation effect prediction and expression profile generation benchmarks, Stack’s zero-shot performance surpasses all evaluated strong baselines in 28 of 31 cases (Kernfeld et al., 2023; Adduri et al., 2025; Luecken et al., 2022). We applied Stack’s unique capacity to create the first perturbed human whole-organism atlas Perturb Sapiens, spanning 28 tissues and 40 cell classes under 201 drug and cytokine perturbations. Perturb Sapiens reveals realistic cellular responses across whole-organism cell types. The cell-type- and tissue-specific perturbation effects in Perturb Sapiens were validated using available in vitro cytokine stimulation datasets. 

# 2. Results

# 2.1. Stack leverages cellular context to learn cell representations that generalize across datasets and tasks

Stack is a large-scale self-supervised encoder-decoder model architecture designed to learn fundamental dependencies across cells and genes from single-cell data collections. We pre-trained Stack on human scBaseCount (Youngblut et al., 2025), the largest available human single-cell data collection that contains 189 million cells after strict quality control (Methods). The training dataset comprised 19,978 SRX samples and 149 million cells, with the remaining 20% of the data held out for validation and testing (Fig. 1A, Methods). Our high-quality training set is over four-fold larger than those of scGPT and Geneformer, and more than three-fold larger than that of UCE (Cui et al., 2024; Theodoris et al., 2023; Rosen et al., 2023). To accelerate model training, we developed a highly efficient h5py-based dataloader that reads consecutive chunks for each single-cell data sample and caches cell index sets (Fig. 1A, Methods). The dataloader achieves high input-pipeline throughput (around $1 . 6 \times 1 0 ^ { 4 }$ cells/second, over 75x faster than a similar model (Adduri et al., 2025)), sufficient to saturate GPU compute and complete pre-training in 2–3 days on a single H100 GPU. 

The input to Stack is a collection of cells, or a cell set, from a single experimental sample. We define each cell’s context as the remaining cells in its set. Stack makes use of a rectangular mask pre-training task that prevents simple imputation shortcuts and enforces single-cell level resolution. Within a mini-batch, a randomly sampled list of genes are masked for all cells. The model is trained to reconstruct gene expression distributions for each individual cell. The mask ratio is sampled from a uniform distribution to enhance feature learning (Dong et al., 2025) (Fig. 1A). After pre-training, Stack outputs cell-level embeddings and gene expressions for new single-cell datasets, empowering numerous applications in both observational and perturbational biology in a zero-shot manner (Fig. 1A), without the need for test-data-specific fine-tuning. Stack gains strong zeroshot power through its inference-time learning capacity, which is detailed later. 

Stack abstracts the latent state of each cell as an ensemble of token vectors, which we term “gene module tokens”. Tokens are generated by projecting gene expression vectors into a latent space using a single-layer perceptron, producing a fixed number of tokens per cell. This tokenization module is trained end-to-end alongside the rest of the model, without relying on external gene semantic information. To our knowledge, Stack represents the first single-cell foundation model to introduce trainable tokenization at the gene-group level. Because the number of tokens (100) is substantially smaller than the number of genes in the data, the model must implicitly learn meaningful gene groupings to preserve biological information. This also yields substantial scalability gains over gene-level tokenization. 

A key innovation of Stack is a new tabular transformer block that enables both intra-cellular and intercellular information flow within the cell set (Fig. 1B). Each block stacks an intra-cellular multi-head attention (MHA) layer, an inter-cellular MHA layer, and a token-wise feedforward network (FFN) layer. In the intracellular MHA layer, the attention mechanism is performed on the gene module token sequence independently for each cell. In the inter-cellular MHA layer, the attention mechanism is on the cell set, with “cell tokens” defined as the concatenation of all gene module tokens. Our design draws inspiration from emerging tabular learning architectures such as TabPFN and TabICL (Hollmann et al., 2025; Qu et al., 2025) and additionally accounts for attention between different gene modules across cells. The final-layer gene module tokens are concatenated to form the cell embedding, and a cell-wise multilayer perceptron (MLP) decoder models the observed gene expression as a probabilistic function of this embedding. In addition to the masked gene reconstruction objective, Stack incorporates a distributional regularization that enforces cell embeddings to decompose into a per-cell-set constant plus standard normal distributed samples (Methods), matching the linear identifiability condition for non-linear latent variable models (Khemakhem et al., 2020; Dong et al., 2024). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/c870158e532c53a87c462f67d54cbe5315b3b7f8b86cc4bcebcc44dab3ccb3f7.jpg)



Figure 1 | Stack: A single-cell foundation model that leverages cellular context. A. Overview of the Stack model. Stack takes scBaseCount human single-cell data (Youngblut et al., 2025) as input, comprising 149 million cells from 19,978 SRX files after preprocessing and filtering. Each file is chunked into consecutive cell sets of fixed size as model input. During pre-training, each input cell set is corrupted by masking a random subset of genes across all cells with a variable masking ratio (Dong et al., 2025). After pre-training, Stack enables zero-shot embedding analysis and a wide range of downstream tasks guided by designed prompts. B. Pre-training of Stack. Stack employs a single-layer multi-layer perceptron (MLP) to project cells into a set of gene module tokens. The tokens are passed through a tabular attention architecture that iteratively applies multi-head attention (MHA) along both gene and cell dimensions, followed by a feedforward network. After ?? tabular attention blocks, the final tokens are concatenated and flattened into one-dimensional vectors per cell (embeddings), and a decoder projects the embeddings back to expression space (Methods). C. Inference-time learning of Stack. The model takes either a single query cell set or a concatenation of prompt and query cell sets as input. Stack performs in-context learning through the tabular attention architecture, and can output prompt-conditioned gene expression. D. Effect of unique cell numbers on validation reconstruction loss under different Stack settings. All Stack models presented here were trained on the scBaseCount subset using identical training and validation sets (Methods). E. Heatmap of adjusted ??-values from Gene Ontology (GO) biological process gene set enrichment analysis, showing the top 10 highest-importance genes (ranked by tokenization weight magnitude) for each Stack (Large) token after pre-training on full human scBaseCount. Module and pathway names, along with additional plot details, are provided in Fig. S2.


During pre-training, this architecture captures the dependency between each cell and its context (the remainder of the cell set), enabling greater control and refinement of predicted single-cell expression. Following a post-training procedure, it also allows for the modification of a cell set of interest (query set) using an artificially designed set of prompt cells that implicitly encode auxiliary conditioning to guide the final model output. This output includes both embeddings and predicted expression values for each target cell (Fig. 1C). The framework offers two key advantages at inference: (1) The dependencies across cells in a set, encoded in intercellular attention layers during pre-training, generalize to unseen datasets to improve zero-shot performance without model updates; (2) It provides a backbone for in-context learning, enabling “cell prompt engineering” at inference. Specifically, the context can be altered to achieve desired outcomes for the query cells, such as shifting gene expressions to match a new donor or predicting the effects of perturbations. Cell prompts can be obtained from any single-cell observational or perturbational dataset, providing high-quality representations of diverse condition signals (such as disease state, genetic/chemical perturbations, donor variability, age) in a unified transcriptomic space. Simulating query data in the prompt context enables both generalization to new biological contexts (cell types, donors etc.) as well as novel predictive tasks (such as perturbation, age etc.) not encountered during training. Both rely solely on prompt-provided information at inference time. 

For model evaluation, we also trained Stack on a version of the CELLxGENE collection (Program et al., 2025) that contains 73.7 million human cells, and a 60-million-cell subset of human scBaseCount in addition to the full scBaseCount. We observed a scaling behavior in terms of various validation metrics across configurations of Stack from 69 to 629 million parameters (Fig. S1A). Increasing hidden dimensionality results in an overall improvement of validation performance. Increasing network depth yields similar validation loss but improves performance on other metrics for the full scBaseCount dataset, while showing mixed effects on the scBaseCount subset (Fig. S1B). Scaling cell set size to 256 optimizes validation loss, while validation reconstruction metrics peak at a cell set size of 128 among the tested values (Fig. S1C). An ablation study confirmed that inter-cellular attention and latent space regularization in Stack both improve validation metrics (Fig. S1D). To assess the impact of informative cell context, we varied unique cells per cell set while holding total context size constant through repetition. Stack outperforms the ablation model without intercellular attention once unique cell numbers in the set exceed 32. The larger Stack model achieves lower validation loss, with gains widening as unique cells increase, indicating enhanced information aggregation capacity (Fig. 1D). 

Finally, Stack tokenization yields highly specific gene groups. Among the top-10 genes per module (ranked by tokenization weight magnitude) , 526 of 699 (75.3%) appear in exactly one gene-module token, even though we impose no explicit sparsity objective. Gene set enrichment analysis confirmed that these tokens are functionally coherent (Fig. 1E, S2). 

# 2.2. Stack generates superior embeddings by learning from cellular contexts at inference time

To evaluate the capabilities of Stack embeddings for individual samples on downstream tasks (Fig. 2A), we developed a comprehensive benchmarking framework that assesses the impact of cellular context and the quality of single-cell level representations, through probing and integration evaluations (Fig. 2B-C, Methods). The datasets for evaluation include: 1) five collections of observational samples, four representing distinct tissues (Kidney, Lymph Node, Brain, and Lung) drawn from a large number of donors (38–223), and Tabula Sapiens (De Boer et al., 2021; Li et al., 2025; Salcher et al., 2022; Gabitto et al., 2024; Consortium* et al., 2022), 2) four large-scale perturbation datasets covering three major perturbation modalities (Drug: OpenProblems, Tahoe-100M; Signaling: Parse-PBMC or Parse; Genetic: X-Atlas:Orion or Xaira) (Luecken et al., 2025; Zhang et al., 2025; Parse Biosciences, 2023; Huang et al., 2025). The LUCA dataset was part of the scBaseCount or CELLxGENE training data, whereas the remaining evaluation datasets were not, thus corresponding to a zero-shot setting (see Methods for details). We first evaluated model embeddings on observational single-cell data by applying linear and multi-layer perceptron (MLP) probes to predict metadata across varying levels of subtlety and resolution, ranging from disease and physiological conditions to cell types. Importantly, our probing schemes incorporate carefully designed procedures, including balanced donor cell numbers, donor-level test set holdout, and group-level cross-validation for regularization hyperparameter optimization (Methods). This rigorously tests whether the model captures biological state signatures through self-supervised learning that generalize to held-out donors. 

We compared Stack with a list of methods that generate embeddings in zero-shot, fine-tuned, or trainfrom-scratch settings. These include principal components of highly variable genes (PC HVG), scGPT (Cui et al., 2024), UCE (Rosen et al., 2023), State (State Embedding or SE) (Adduri et al., 2025), TranscriptFormer 


A


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/e5b297c7daa3370f0b6f424d41a215786a135fe2c33b18089cb9626a8205d233.jpg)



B


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/d591638842f440058f3356a2196767361d663f0a4dcfcebd0489b8b5afd0a1f3.jpg)



C


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/4abebe04f91e544d5ddd6e5f566d24ac1eace8275656fbbe4fd430962d348733.jpg)



D



Disease and physiological condition probing performance on observational scRNA-seq data


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/e84ade10a864e639576018ebefbb2c0da8103473db5aa19420a5f99d87c6ca8d.jpg)



E



Perturbation classification performance on perturbational scRNA-seq data


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/b3f306bb3a4f351fce77d0502d7a8780367af6db4e2934471f4d06e857392703.jpg)



F



Integration performance on Tabula Sapiens per tissue


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/5f6173780acb28094c55fa49d9aef50ac142b7f6320c712b1b9859ad8a5cdf52.jpg)



Figure 2 | Evaluation of Stack embeddings. A. In the evaluations presented here, Stack serves as a context-aware embedding model, processing query cell sets from individual samples. B. Schematic illustration of the probing evaluation framework. C. Schematic illustration of the integration evaluation framework. D. Per-cell-type linear probing results. Probing scores represent the balanced accuracy (for classification tasks) or Pearson r (for regression tasks), which is first normalized for each cell type to a [0, 1] scale across all methods, and then averaged across all cell types. The number of experiments per dataset (equal to cell types): n=12, 20, 10, 20. AKI: Acute kidney injury. CKD: Chronic kidney disease. BCL: B-cell lymphoma. COPD: Chronic obstructive pulmonary disease. LUAD: Lung adenocarcinoma. LUSC: Lung squamous cell carcinoma. NSCLC: Non-small cell lung cancer. Other categories: Kidney (hypertension, diabetes history), Brain (Microinfarct pathology, ADNC, Braak stage, Thal phase, CERAD score, APOE4 status), Lymph node (LymphoMAP), LUCA (UICC stage, ever smoker). ADNC, Alzheimer’s Disease Neuropathologic Change. UICC, Union for International Cancer Control. E. Linear probing evaluation on classifying perturbations from 4 perturbation atlases. The number of experiments (evaluated cell types) per dataset: n=6, 60 (20 per plate), 17, 2. F. scIB evaluation for each tissue in Tabula Sapiens (Consortium* et al., 2022). All Stack results presented here are based on one model with the (Large) setting pre-trained on full human scBaseCount.


(Pearce et al., 2025), scVI pre-trained on the scBaseCount subset and fine-tuned on the target dataset (scVI FT) (Lopez et al., 2018; Gayoso et al., 2022), and scVI trained from scratch on the target dataset (scVI from scratch). In per-cell-type linear probing, Stack shows a substantial advantage compared with alternative methods across disease/other categories on the four observational datasets (Fig. 2D, Fig. S3A). The only exception is the classification of other categories in LUCA, a dataset partially included in the training sets of both Stack and State (SE), where Stack ranks second and underperforms State (SE) by 2.2%. To control for the information flow across cell types, we constructed a new cellular context setting where the cells included in each cell set are constrained to be of the same cell type. Stack still achieves the best overall performance among all methods, albeit by a smaller margin (Fig. S4). Removing this cellular context information by shuffling the cell order results in a notable decrease in performance, with final results similar to those of the alternative methods (Fig. S4). To rule out the possibility that Stack ’s advantage arises from simple information leakage from more informative cell types, we further evaluated probing performance in the overall best-performing cell type across all methods, and Stack ’s advantage remained (Fig. S5). The association between Stack embedding performance and cellular context configurations suggests that Stack’s improvement stems from its ability to extract information from novel cellular contexts. 

A similar advantage of Stack is observed with MLP probing across all four observational datasets (Fig. S3B). In either linear or MLP probing, all other advanced foundation models do not show consistent advantage over baseline methods such as PC HVG and scVI trained from scratch on the dataset of interest (Figs. 2D, S3, S4). In terms of cell type classification, all methods demonstrate highly similar performance (Fig. S3C). The small variation in performance is likely due to the cleaner cell type signal and its manually annotated nature. The competitive performance of Stack suggests that the context-aware mechanism does not compromise Stack’s representation at single-cell resolution. 

We next evaluated different models’ performance in classifying chemical, signaling, and genetic perturbations (Fig. 2E). Stack outperforms existing methods on all perturbational datasets tested and is the only one to consistently outperform scVI trained from scratch. Notably, Stack shows substantial improvements in discriminating between perturbation effects in large-scale datasets Tahoe and Parse, outperforming the best alternatives by approximately 100% despite being trained almost exclusively on observational data. All methods show low absolute performance in classifying genetic perturbations in the Xaira dataset, likely due to measurement noise and gene expression similarity across perturbations. These results suggest that aggregating context information is crucial for accurate predictions of subtle biological states. 

We also evaluated cell type preservation and donor label correction performance of different embeddings on observational data through scIB integration metrics (Luecken et al., 2022). Across all four observational datasets, Stack ranks as the top performer, surpassing the best alternative (State (SE)) by +1.8% (Fig. S6A). In Tabula Sapiens, Stack outperforms alternative methods in 21 of 25 tissues (ranking second in eye, heart, mammary and uterus, where scVI from scratch performs best), demonstrating superior performance across human tissues (Fig. 2F). This batch integration ability is emergent as it is not explicitly enforced during training. Scaling Stack training data and model size each lead to improvements in batch integration performance (Fig. S6B). UMAP visualization supports Stack’s effectiveness in clustering fine-grained cell states and integrating batch-level information (Fig. S6C). The results remain similar across model scales and extend to dataset label integration evaluations (Fig. S6D-E). These results establish Stack as a powerful embedding model that enhances zero-shot prediction and integration by leveraging cellular contexts. 

# 2.3. Stack enables in-context learning of novel predictive tasks with cells after post-training

During pre-training, Stack is exposed to sets of cells from the same biological sample (such as donor or experimental condition), limiting the base model’s utility for tasks where the context is engineered by the user to produce a desired cell state. This limitation parallels the necessity of supervised fine-tuning (SFT) and reinforcement learning (RL) for adapting pre-trained large language models to follow user instructions (Wei et al., 2022; Longpre et al., 2023; Ouyang et al., 2022). To teach Stack to follow instructions, we define a cell conditioning task that involves two cell populations: prompt and query. The prompt cells specify the desired biological state or condition, while the query cells specify the cell type of interest. The objective is to predict counterfactual states of query cells, i.e., their gene expression profiles under the prompt condition, encompassing tasks such as generalizing perturbation effects to new cell types and across datasets. Here, prompt and query cells can come from different datasets, comprise non-overlapping cell types, and their annotations may be unavailable. Therefore, a supervised approach is not feasible for this task, and a foundation model with zero-shot capabilities is crucial. The context-awareness of Stack makes it particularly suitable for these tasks. 

We developed a novel post-training recipe via self-distillation to adapt the Stack model for the task. This post-training process is closely related to masked language diffusion models (Sahoo et al., 2024), whose training objective is to recover masked tokens within input sequences with a masking ratio ranging from 0 to 1. In this procedure, each cell set from a single biological sample is grouped by type and partitioned into two subsets: prompt cells, which remain visible, and target cells, which are held out, analogous to unmasked and masked tokens in masked diffusion models. During post-training, the target cells are replaced with typematched query cells drawn from a different biological sample. Stack is then post-trained to reconstruct the held-out target cells from query cells, conditioned on the prompt cells. Through this post-training procedure, Stack learns to predict counterfactual states for any cell population conditioned on the prompt cells, enabling conditional cell state generation. 

We use a pre-trained Stack model as a teacher to extract embeddings for the target cells (Fig. 3A). The student Stack model is optimized to predict target cell distributions in both embedding and gene expression spaces, with additional regularization terms. The teacher model is updated using exponential moving average of the student model’s parameters, allowing it to progressively adapt to new data distributions while preserving knowledge acquired during pre-training. To compute the distributional match in gene expression space as a training objective, we employ a zero-inflated normal distribution approximation that enables reparameterization (see Methods). Finally, a multi-layer perceptron (MLP) classifier is trained to classify cells in the embedding space, where a lower score indicates greater similarity to the prompt condition and thus higher confidence in generation quality. At inference time, this score guides an iterative refinement procedure (Fig. 3A, Methods). 

After post-training, we can use Stack as a conditional generative model to simulate novel cell populations. In the generative procedure, Stack receives cell sets containing concatenated prompt and query data. Stack predicts the gene expressions on all query positions, as well as their scores using the MLP classifier. At each iteration, we replace a fraction of highest-confidence query cells’ gene expression values with model predictions. This process is conducted in an iterative manner and finishes when the fraction reaches zero, at which point all query cells are replaced with predictions. Throughout the iteration, the fraction of prompt data in the input cell set is gradually increased to enable finer control (Methods). 

For post-training data, we curated a large 55-million-cell scRNA-seq data collection comprising a set of large CELLxGENE datasets (>50,000 cells, >5 donors) and the Parse PBMC 10M dataset, which contains 12 donors and 90 cytokine perturbations (Fig. 3B). This training data emphasizes in vivo cell types, with particular focus on immune cells. To efficiently post-train on our large data collection, we developed an extended post-training dataloader that maximizes cell-type-aware local chunking. We evaluated the post-trained model on four downstream cell prompting/in-context learning tasks spanning three categories: perturbational, observational, and hybrid ICL tasks (Box 1). 

Our evaluation leverages key metrics proposed in cell-eval (Adduri et al., 2025), which can be categorized into pseudo-bulk correlation metrics (Pearson Delta, DE Spearman LFC) and differential expression (DE) metrics (PR AUC, DE overlap accuracy, Spearman effect size; see Methods). We also report the Jaccard similarity metric, which measures the overlap between two DE gene sets normalized by their union, thereby adjusting for both predicted and ground-truth DE gene set sizes (Fig. S7). DE direction match and DE precision-at-N serve as auxiliary metrics when pseudobulk or DE metrics are less appropriate or not applicable (Methods). For setting 4, where target and query belong to different datasets, we additionally evaluated scIB batch integration metrics. Alternative baselines include the query data itself (input baseline), the nearest/same cell type in the prompt sample, State (Adduri et al., 2025), and two strong baselines identified in Adduri et al. (2025) (PerturbMean/DonorMean, scVI). For perturbational ICL tasks, we employ a “synthetic control” approach that uses the unperturbed version of the prompt sample to additionally predict a control profile, which then serves as the reference for the perturbation effect predictions when computing cell-eval metrics. The same syntheticcontrol procedure is applied to the closest/same-cell-type baseline and scVI, resulting in stronger baselines. For the remaining tasks, Stack operates without auxiliary samples, whereas DonorMean and scVI baselines still require them; we therefore designate these baselines as “oracle” in those cases. The evaluation data includes seven datasets that were never seen during either Stack pre-training or post-training: 1. OpenProblems drug perturbation (Luecken et al., 2025), 2. Cytokine stimulation (Dong et al., 2023), 3. Immune aging (Wells et al., 2025), 4. Tabula Sapiens (Consortium* et al., 2022), 5. Kidney atlas (De Boer et al., 2021), 6. Lymph 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/29b9f52b803c77b42d041c7afd90cff7689564c58bf634b3c245c4ef87bd68c3.jpg)



Figure 3 | Post-training of Stack for in-context cell prompting tasks. A. Schematic illustration of the Stack posttraining framework. Cell sets are organized by cell type and partitioned into prompt and target cell sets. During posttraining, target cells are replaced with type-matched cells from different conditions (queries) to serve as model input. The model learns to predict gene expressions and embeddings of target cells via distributional alignment and self-distillation, with an MLP classifier guiding inference-time generation. B. Overview of post-training data. Training data comprised approximately 55M cells from curated CELLxGENE datasets and the Parse PBMC dataset (12 donors, 90 perturbations). C. Evaluation of perturbation effect prediction across cell types on the Dong et al. (2023) cytokine perturbation dataset (6 cytokines). In the first four panels, each point represents the average result of a cytokine condition. In the last panel, each point represents a cell type (B cells, myeloid cells, T cells). Percentages in titles represent the average improvement of Stack over the best non-Stack baseline. All Stack (zero-shot) predictions were generated using a post-trained Stack (Large) model with a mask diffusion procedure (?? = 5). D. Pearson Delta evaluation results for Dong et al. (2023), stratified by cytokine. (Continued on next page)


Figure 3 (Continued) | E. Evaluation of perturbation effect prediction across cell types on the OpenProblems drug perturbation dataset (Luecken et al., 2025) (12 drugs). Points represent drug conditions (first four panels) or cell types (last panel). F. Evaluation of T cell response prediction across samples. Each donor from Parse PBMC serves as a prompt, and donors a/b from (Dong et al., 2023) serve as queries. Each point represents the average result of a cytokine condition (7 single/combinatorial conditions). G. Evaluation of donor-specific gene expression generation across four atlases (Kidney: (De Boer et al., 2021); Lymph Node: (Li et al., 2025); Liver: (Edgar et al., 2025); PBMC: (Wells et al., 2025)). Nonoverlapping cell types from sampled donor pairs serve as prompts and queries. Each point represents one evaluation dataset. H. Evaluation of condition-specific expression generation across four PBMC atlases. Drug perts: 15 conditions with all PBMC cell type expression profiles available in (Luecken et al., 2025); Aging: first 10 donors in (Wells et al., 2025); Parse donors: all control PBS conditions from 12 donors; Parse perts: first 20 perturbation conditions from donor 1. Immune cells from Tabula Sapiens (Consortium* et al., 2022) serve as queries for all cases. Each point represents one evaluation case. For all scores, higher values indicate better performance. 

node BCL (Li et al., 2025), 7. Liver atlas (Edgar et al., 2025). Additionally, we include the Parse PBMC dataset (Parse Biosciences, 2023) as prompts (not as queries) in our evaluations. 

# Box 1. In-context learning (ICL) tasks for cell prompting.

• Perturbational ICL (prompts are perturbed cells and queries are control cells) 

1. Perturbation effect prediction in novel cell types from same sample: We use randomly sampled perturbed cell types as the prompt, and control cells from remaining cell types as the query. Stack simulates the prompt specified perturbation condition in the query cell types (Fig. 3C–E). 

Example: Given a PBMC sample where only T cells were perturbed with IL-6, we use perturbed T cells as the prompt and unperturbed B cells/monocytes as the query to predict IL-6 effects on B cells/monocytes. 

2. Perturbation effect prediction in novel samples: We use one single perturbed cell type (T cells here) as the prompt, and control T cells from another donor in a different dataset as the query. The setting assesses the model’s ability to predict responses observed in the query sample (Fig. 3F). 

Example: We collected a new PBMC sample and want to predict how T cells respond to IFN-?? stimulation. We use IFN-??-perturbed T cells from a published reference dataset as the prompt, and control T cells from our new sample as the query. 

• Observational ICL (prompts and queries are cells from observational scRNA-seq samples) 

3. Hold-out cell type prediction: We use sampled cell types from one donor as the prompt and non-overlapping cell types from another donor in the same dataset as the query. Stack predicts the expression profile of query cell types in the prompt donor (Fig. 3G). This setting evaluates the model’s ability to capture donor-specific expression differences in observational datasets. 

Example: In a patient cohort, Patient A has profiled macrophages and T cells but lacks fibroblasts due to tissue availability. We use Patient A’s cells as the prompt and fibroblasts from Patient B as the query, to impute Patient A-specific fibroblast expression. 

• Hybrid ICL (prompts and queries are from different observational or perturbation studies) 

4. Cross-dataset cell type generation: We use sampled cell types from one condition as the prompt and non-overlapping cell types from another dataset as the query, and predict the query cell type profile in the prompt sample context (Fig. 3H). This setting assesses the model’s capacity to generate cell types absent from an observational atlas or perturbation data of interest. 

Example: Using drug perturbed T cells and B cells as the prompt, and dendritic cells from an independent scRNA-seq atlas as the query, we predict perturbation responses in dendritic cells that were never experimentally perturbed. 

In setting 1 (perturbation effect prediction across cell types) for the Dong et al. (2023) cytokine stimulation dataset, Stack not only exhibits an advantage across all metrics (Fig. 3C, S7), but generalizes the global effect of subtle cytokine perturbations, including IL-6 and TNF-?? across cell types, as demonstrated by Pearson Delta (Figs. 3D). The overall advantage also extends to the OpenProblems drug perturbation dataset, which comprises drug conditions unseen during model pre-training or post-training (Fig. 3E, S7). In setting 2, Stack outperforms alternative methods (including perturbation effects of observed prompt T cells) in DE metrics but performs similarly in pseudobulk metrics, across both individual and combinatorial cytokine conditions. Nevertheless, because Parse and Dong et al. (2023) use different stimulation protocols in dosage and stimulation duration, generalizing subtle effects of several cytokines (TNF-??, IL-6) alone yields near-zero pseudo-bulk and DE scores across all methods (Fig. 3F). 

In setting 3, oracle DonorMean shows the strongest performance in pseudobulk correlation metrics, while Stack maintains its lead in directional and DE metrics across four evaluated tissues (Fig. 3G, S7). Notably, Stack demonstrates particular strength in generating cell types across datasets (Setting 4), as evidenced by scIB integration and cell-eval metrics (Figs. 3H, S7). This advantage may stem from the context-aware capacity acquired during Stack pre-training, as supported by the suboptimal performance of Stack trained from scratch on perturbational ICL tasks (Fig. S8). 

As a zero-shot approach, Stack achieves the strongest overall performance across all ICL tasks, ranking first in 28 of 31 evaluations (Fig. 3C-H, S7). Closest/same-cell-type prompt cells emerges as a strong baseline when evaluated on cell-eval metrics, which has not been sufficiently addressed in previous perturbation prediction studies. Despite being trained on both prompt and query data, State is outperformed by Stack across all metrics on the evaluated perturbation tasks. This is likely because State was designed for settings with orders of magnitude more supervised data than those evaluated here. Stack’s success in low-data, crossexperiment settings demonstrates its utility as a foundation model when supervised fine-tuning is insufficient, for example, when interrogating cell types that are difficult to perturb experimentally. The multi-step mask diffusion generative procedure shows modest advantages over Stack one-step prediction and alternative generative schemes across ICL tasks (Fig. S9). The competitive performance of Stack on unseen prompts and queries (Dong et al., 2023; Wells et al., 2025; Consortium* et al., 2022), novel perturbations (Luecken et al., 2025), and cell types and tissues beyond peripheral blood (De Boer et al., 2021; Li et al., 2025; Edgar et al., 2025; Sikkema et al., 2023) underscores the generalizability of our approach to previously unencountered datasets and diverse biological tasks. 

# 2.4. Stack generates a virtual whole-organism perturbational atlas

We employed Stack to generate a perturbational whole-organism atlas (Perturb Sapiens) through ICL, using 90 cytokine perturbations in Parse and 111 drug perturbations in OpenProblems as the prompt and the entire profile of Tabula Sapiens, after tissue balancing, as the query (Fig. 4A) (Parse Biosciences, 2023; Luecken et al., 2025; Consortium* et al., 2022). UMAP visualization demonstrates a single-cell resolution map with tissuespecific and cell-type-specific expression for Perturb Sapiens (Figs. 4B, S10A). Inspection of the MLP classifier scores suggests a variation in generation confidence for different cell and tissue types (Figs. S10B–C). The classifier assigns low confidence (high logit value) to several rare cell types such as transitional epithelial cells in both drug and cytokine example perturbations. We restricted our subsequent analyses to cells with logits smaller than a threshold (2.5) suggesting high confidence. 

As a representative example, we inspected the effect of IFN-?? perturbation versus control in Perturb Sapiens to assess model generalization beyond the effects observed in the prompt or prompt-related (immune) cell types. Stack generates a highly realistic differential expression map with cell-type specificity (Fig. 4C). The top differentially expressed genes exhibit near-perfect concordance between prompt and generated immune cells. Notably, although only a single donor from Parse was used as the prompt, Perturb Sapiens achieved stronger alignment with the aggregated response across all 12 Parse donors, demonstrating its capacity to overcome individual experimental noise (Fig. 4C). Known downstream targets of IFN-?? were activated across a variety of immune and non-immune cell types (e.g., IFIT3, ISG15, CXCL10, CXCL11, IDO1). In non-immune cell populations, IFN-?? induced CIITA in stromal and contractile populations while broadly suppressing extracellular matrix and adhesion genes (LUM, FBLN1, LAMA4, ITGA8); this was accompanied by vascular remodeling signatures (EMCN, ANGPT2, CEACAM1, PLAT) consistent with IFN-driven inflammatory reprogramming. These genes are either not differentially expressed in the Parse PBMC data or exhibit inconsistent expression trends between the prompt donor and the aggregated profile. This indicates that Stack successfully generalized beyond the prompt data to capture immunomodulatory and remodeling responses specific to nonimmune lineages. Dactolisib perturbation leads to global repression of interferon-stimulated genes (ISGs) (Fig. S11), aligning with the known role of dactolisib as a PI3K/mTOR inhibitor and previous analysis (Dong et al., 2024). Additionally, non-immune lineages showed selective remodeling, with cytoskeletal/adhesion and 


A


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/eae4dc6c16f149093a7dcec08204c73cbf48ea940ba0a17824d6a94e6c4e8bcc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/3e7362e9db9b40fa73fb7b6297207e4cca6fc3bd9891059d81d5e0f4940e5d60.jpg)



C


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/3328bc687171e878363320e41d0de4392a6e8ef9b5b4eb3b3a0a7b508ce26797.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/ad63d2b12c5c017781a1223d22f8cefd85f3748e05788418c7a2ad849d310489.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/09915f87cf59692020c352607cd7ad6c0a8bd23af24c5e0d6e5e9d70bead46fa.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/31728a5907a4c005ae7178550d1133b6631bc4722e2e84ec8ba85b90b26d4f05.jpg)



Figure 4 | Analysis of a perturbational whole-organism atlas Perturb Sapiens. A. Overview of Perturb Sapiens. We utilize PBMC perturbation datasets (90 cytokines from Parse (Parse Biosciences, 2023); 144 drugs from OpenProblems (Luecken et al., 2025)) as prompts and Tabula Sapiens (Consortium* et al., 2022) as queries (left). For each tissue and cell type, Perturb Sapiens comprises simulated gene expression profiles under drug and cytokine perturbations. The resulting atlas spans 28 tissues and 513,870 cells per perturbation condition (right) (Consortium* et al., 2022). B. UMAP visualization of an example Perturb Sapiens generated by combining cytokine ADSF with Tabula Sapiens, colored by tissue and cell class. C. Log2-fold-change heatmap comparing IFN-?? Perturb Sapiens to control. Significantly differentially expressed genes are shown in color; non-significant genes are shown in gray. D. Evaluation of Perturb Sapiens epithelial interferonbeta (IFN-??) effects using single-cell IFN-?? stimulation data from primary airway epithelial cells (Koh et al., 2023). E. Evaluation of Perturb Sapiens epithelial interleukin-13 (IL-13) effects using single-cell IL-13 stimulation data from primary airway epithelial cells (Koh et al., 2023). F. Evaluation of Perturb Sapiens epithelial interleukin-1 beta (IL-1??) effects using bulk IL-1?? stimulation data from primary keratinocytes (Swindell et al., 2018).


stress-response programs being modulated (e.g., TUBA1C, FHL1, LAMA2, NINJ1, MT1X). 

While a majority of drug perturbations in OpenProblems only contains the T cell lineage, Stack also generates high-quality predictions across all cell types as observed for proscillaridin-A and ketoconazole (Fig. S12–S13). Proscillaridin-A elicited a broad pro-inflammatory, innate-like activation program across both immune and non-immune cell types. Non-immune lineages exhibited distinct transcriptional remodeling involving vesicular trafficking and membrane dynamics (RAB30, STX3, LYST) alongside lineage-specific regulators (CPEB4, PLAGL1, TSC22D1). Notably, ketoconazole exhibited pronounced donor-specific effects across the OpenProblems dataset. Perturb Sapiens effectively captures these individualized responses in the prompt donor while maintaining alignment with bulk expression patterns for a number of genes where donor-specific differential expression is not detected (Fig. S13). Overall, top DEGs in drug perturbation conditions show strong concordance between prompt and generated immune cells comparable to the cytokine case, although the drug Perturb Sapiens yields additional positives where prompt immune cells also express predicted nonimmune DEGs. For quantitative validation, we benchmarked Perturb Sapiens immune cells against biological replicates from the same drug/perturbation datasets, assessing their ability to capture prompt perturbation effects. Although Perturb Sapiens yields low Pearson Delta scores likely due to batch effects, it achieves superior DE overlap accuracy compared to biological replicates, and the performance of Perturb Sapiens remains consistent across both drug and cytokine perturbations (Fig. S14). These findings support Stack’s ability to generate biologically meaningful, cell-type-specific responses to diverse perturbations, without observing any perturbations in the query cell type. 

We next quantitatively evaluated Stack’s performance on generating perturbed non-immune cells, focusing on the epithelial lineage due to the prevalence of in vitro epithelial cytokine stimulation datasets (Koh et al., 2023; Lee et al., 2022; Swindell et al., 2018; Saito et al., 2021). We benchmarked the model’s ability to reproduce the effects of five cytokines (type I IFN, IL-13, IL-1??, TNF-??, IL-17A) in single-cell or bulk data using either direct prediction of each cytokine or functionally similar cytokines, applying the same set of cell-eval pseudobulk and DE metrics as before. Stack demonstrated strong performance in generating epithelial-specific type I IFN responses, outperforming both generated immune cell types and Parse immune cells used as the prompt (Fig. 4D). Notably, the model exhibited a clear preference for airway epithelial cells over other epithelial subtypes, consistent with the in vitro experiment. Unlike IFN-??, IL-13 is a cytokine that primarily affects epithelial cells and has weaker effects on immune cells. Despite this limited signal in the prompt, Perturb Sapiens demonstrates cell-type-specific alignment with the in vitro airway epithelial experiment, albeit with overall lower scores (Fig. 4E). This cell-type specificity generalizes to other tissues, as observed in the IL-1?? keratinocyte stimulation experiment (Fig. 4F). Across all three cases, Perturb Sapiens achieves better performance than the prompt data, with scores ordered by cell-type similarity. Evaluations without confidence filtering results in reduced cell-type specificity, confirming the essence of the procedure (Fig. S15). 

For IL-17 and TNF-??, both the Parse prompt and generated data showed negative association with epithelial response (Fig. S16), with latter replicated in an independent TNF-?? epithelial stimulation dataset (Table S1). Upon further investigation, we found that despite concordance in several TNF-?? response genes, including metallothioneins and cell adhesion molecules, TNF-?? suppressed the expression of genes involved in NF-??B signaling and type I ISG programs in both Parse T cells and Perturb Sapiens, while inducing these signatures in the in vitro epithelial dataset (Table S2). This phenomenon aligns with previously documented secondary signaling effects of TNF-?? (Kalliolias and Ivashkiv, 2016). In summary, our results suggest that Stack can simulate perturbed non-immune cells with cell-type and tissue specificity, with alignment to ground truth data correlating with the perturbation’s effect size and specific biological mechanisms. 

Finally, we computed log-fold-changes and statistical significance of perturbation effects across all available conditions in Perturb Sapiens, stratified by cell type and tissue. This framework enables multi-scale characterization of perturbation responses and provides a unified approach to categorizing drug and cytokine perturbations. At the global scale, predicted perturbation similarities clustered coherently by cell lineage and tissue proximity (Fig. S17). At the local scale, we decomposed the concatenated perturbation effect space for each cell type and tissue using independent component analysis, revealing diverse response modules that segregated by perturbation type without evident batch effects across drugs and cytokines (Fig. S18A-C). Examination of the top contributing genes within each independent component identified IFN/inflammatory signaling as the dominant response axis, with additional modules reflecting stress response and ECM remodeling pathways (Fig. S18D-F). Collectively, Perturb Sapiens provides a comprehensive, multi-resolution view of perturbation effects across cell types and tissues, representing a rich resource that extends well beyond the 

analyses presented here. 

# 3. Discussion

As large atlases of single-cell transcriptomic profiles are compiled across tissues, species, and diseases, foundation models present an exciting opportunity to learn universal biological principles and patterns that generalize beyond experimentally observed data (Bommasani et al., 2021). A virtual atlas of cell states could significantly expand our understanding of cell biology through uncovering cellular states that are difficult to probe experimentally but can be inferred through relationships learned from existing data (Bunne et al., 2024; Roohani et al., 2025). However, realizing this promise requires models that transfer robustly across conditions and tasks. Most existing models present several limitations: they often fail to generalize to previously unseen conditions, provide no benefit over methods that are specifically fine-tuned on those datasets, and cannot perform new tasks without being explicitly trained to do so. 

Here, we introduce Stack, a single-cell foundation model that leverages information from the cellular context of each cell to create enhanced representations. This design enables Stack to consistently outperform models trained from scratch on each evaluation dataset, an outcome that, to our knowledge, has not been observed for any existing single-cell foundation model, and highlights Stack ’s ability to meaningfully leverage information acquired during pre-training. This dependence on context also enables a novel capability: engineering context to design cell state. Because context can be defined in many ways, such as an applied perturbation, a disease state, or a new donor, Stack supports inference-time learning of new tasks, including zero-shot generalization to new biological contexts and datasets. 

Through this capability of in-context learning, Stack enables a new approach to single-cell modeling in which counterfactual cell states can be generated via prompting with only cells. Importantly, Stack extends existing perturbation modeling by removing the reliance on perturbation labels, cell type encodings, and testsample-specific controls, enabling direct, label-free comparison of cell states to resolve context-dependent responses that transcend categorical annotations. These results position Stack as a generative cellular model capable of predicting unobserved gene expressions across cell types, perturbations, and novel donors, with potential to accelerate therapeutic and drug discovery cycles. As a concrete demonstration, we applied Stack to generate Perturb Sapiens, an organism-wide atlas of perturbed cells spanning 28 tissues, 40 cell classes, and 201 perturbations, a resource we believe will be of broad value to the community. 

Despite its strong performance and novel capabilities, Stack has several limitations that present opportunities for future research. The current model is trained exclusively on human single-cell data. Extending this framework to multi-species applications would require additional design in the tokenization procedure to account for gene misalignment across species. Model calibration for rare cell types and weak perturbation effects remains to be established, marking an important area for future development. Signaling cascades may introduce time-dependent or secondary perturbation effects as we observed for TNF-??, which may complicate interpretation of Stack predictions. Finally, as the model is post-trained primarily on in vivo cell types, especially immune cells, a more sophisticated data curation and alignment scheme may improve model generalization across in vitro perturbation studies and in vivo observational data. 

# 4. Methods

# 4.1. Stack model

# 4.1.1. Generative process

Stack models the state of cell $k , \mathbf { E } ^ { ( k ) } \in \mathbb { R } ^ { n \times d }$ , as a ensemble of ?? token vectors ${ \bf e } _ { i } ^ { ( k ) } \in \mathbb { R } ^ { d }$ , which we term gene module tokens, since each token represents a coherent subset of gene variability in the single cell: 

$$
\mathbf {E} ^ {(k)} = [ \boldsymbol {e} _ {1} ^ {(k)}; \boldsymbol {e} _ {2} ^ {(k)}; \dots ; \boldsymbol {e} _ {n} ^ {(k)} ]. \tag {4.1}
$$

The flattened state vector $\bar { \mathbf { E } } ^ { ( k ) } \in \mathbb { R } ^ { n d }$ is linked to the ground truth gene expression $\boldsymbol { x } ^ { ( k ) } \in \mathbb { R } ^ { G }$ with the generative process considered in scVI models (Lopez et al., 2018; Gayoso et al., 2022). The generative process involves a latent decoding transformation ?? and cell library size scalar $l ^ { ( k ) } \in \mathbb { R }$ . The final expression count is modeled through a negative binomial (NB) distribution with mean and dispersion parameters $( \pmb { \rho } ^ { ( k ) } , \pmb { \theta } ^ { ( k ) } )$ : 

$$
\left(\boldsymbol {\rho} ^ {(k)}, \boldsymbol {\theta} ^ {(k)}\right) := f \left(\bar {\mathbf {E}} ^ {(k)}\right) \in \left(\mathbb {R} ^ {G}, \mathbb {R} ^ {G}\right); \quad \boldsymbol {x} _ {g} ^ {(k)} \sim \mathrm{NB} \left(l ^ {(k)} \boldsymbol {\rho} _ {g} ^ {(k)}, \boldsymbol {\theta} _ {g} ^ {(k)}\right). \tag {4.2}
$$

This generative process may be seen as a hybrid of transformer-based data modeling and biophysical-like models considered in scVI. The most prevalent differences from both strategies are summarized as follows. 

• Stack does not consider low-dimensional latent variables as in scVI models; the total cell token dimensionality ???? $\sim 1 0 ^ { 3 }$ , which is comparable to large language models and large-scale single-cell selfsupervised-learning models (Cui et al., 2024; Rosen et al., 2023; Adduri et al., 2025). 

• Stack does not include structural tokens (e.g. CLS tokens) as in several classic (single-cell) transformer models (Devlin et al., 2019; He et al., 2022; Rosen et al., 2023; Adduri et al., 2025). For downstream fine-tuning or linear probing on embeddings, the concatenation of all output tokens are used as the Stack model embedding output. 

The generative process corresponds to the model decoding procedure from cell state embedding to gene expression, operating independently across cells. 

# 4.1.2. Stack architecture

During pre-training, the Stack model receives a cell set $\ b { X } \in \mathbb { R } ^ { \ b { K } \times \ b { G } }$ , where ?? denotes the number of cells and ?? the total number of genes. The cell set includes cells continuously indexed from the same SRX experiment (for scBaseCount (Youngblut et al., 2025)) or the same dataset (for CELLxGENE (Program et al., 2025), where cells in datasets are primarily sorted by donor ID). The organization of each cell set introduces dependency across cells within the cell set, which serves as rich auxiliary information. Stack encodes the cell set ?? $\in \mathbb { R } ^ { K \times G }$ with the following architecture: 

• Tokenization. First, each cell ?? in the cell set $\boldsymbol { x } ^ { ( k ) } \in \mathbb { R } ^ { G }$ is projected into dimension $n \times d$ with a 1-layer perceptron, where ?? is the number of gene module tokens and ?? denotes the token size. Then a gene token embedding $\pmb { P } \in \mathbb { R } ^ { n \times d }$ is added on the perceptron output. This results in a tensor $\mathbf { Z } _ { 0 } ^ { ( k ) } \in \mathbb { R } ^ { K \times n \times d }$ for the cell set. 

• Tabular transformer layer. The tensors are processed by a stack of $N _ { L }$ tabular transformer layers $( \{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { N _ { L } } )$ : 

$$
\mathbf {Z} _ {i} = \mathcal {T} _ {i} \left(\mathbf {Z} _ {i - 1}\right), \quad i \in \{1, 2, \dots , N _ {L} - 1 \}; \tag {4.3}
$$

$$
\mathbf {E} = \mathcal {T} _ {N _ {L}} (\mathbf {Z} _ {N _ {L} - 1}).
$$

Each layer T?? applies a dual attention mechanism on the cell-set-level representations. In the following description, each attention module is a standard multi-head attention (MHA) block: the input is projected into multiple heads, attention outputs are concatenated and projected back, then combined with the input through a residual connection followed by layer normalization. 

1. Intra-cellular attention. An intra-cell MHA module operates across the ?? tokens independently for each cell in $\mathbf { Z } _ { i } \in \mathbb { R } ^ { K \times n \times d }$ . The sequence length is ??, and the feature dimension is ??. The attention head number is ??????. $N _ { H _ { g } }$ 

2. Inter-cellular attention. An inter-cell MHA module operates across ?? cells, with the $( n , d )$ dimensions flattened into ????. The sequence length is ??, and the feature dimension is ????. The attention head number is $N _ { H _ { c } }$ . 

3. Feedforward network (FFN). A position-wise FFN independently processes each ??-dimensional token, followed by residual connection and layer normalization, yielding $\mathbf { Z } _ { i } \in \mathbb { R } ^ { K \times n \times d }$ . 

• Cell-wise decoder. Finally, the cell state embedding $\mathbf { E } \in \mathbb { R } ^ { K \times n \times d }$ is decoded into gene expression space independently for each cell in $\{ 1 , 2 , \cdots , K \}$ . The decoder $f$ is implemented as a multi-layer perceptron (MLP) that maps the flattened embedding $\bar { \mathbf { E } } ^ { ( k ) } \in \mathbb { R } ^ { n d }$ to the negative binomial parameters $( \bar { \pmb { \rho } } ^ { ( k ) } , \pmb { \theta } ^ { \bar { ( } k ) } )$ ∈ $( \mathbb { R } ^ { G } , \mathbb { R } ^ { G } )$ , following the generative process described above. 

# 4.1.3. Pre-training objective

The model is pre-trained on a masked gene reconstruction task. For each input cell set $X ,$ a random subset of genes $M \subset \{ 1 , 2 , \cdots , G \}$ is selected, and their expression values are masked across all ?? cells. The masking ratio is randomly sampled from $( p _ { \mathrm { m i n } } , p _ { \mathrm { m a x } } ) = ( 0 . 1 , 0 . 8 )$ for each mini-batch. This selection range aims to cover gene dependencies of various strengths, following $\mathrm { R } ^ { 2 } \mathrm { M A E }$ (Dong et al., 2025). We adopt a higher $p _ { \mathrm { m a x } }$ (0.8) than in (Dong et al., 2025) (0.5) as the inter-cellular information brings additional information for implicit denoising. The pre-training objective combines a reconstruction loss with a latent space regularization term: 

$$
\mathcal {L} = \mathcal {L} _ {\text { recon }} + \lambda_ {\mathrm{SW}} \mathcal {L} _ {\mathrm{SW}}. \tag {4.4}
$$

The reconstruction loss, $\scriptstyle { \mathcal { L } } _ { \mathrm { r e c o n } ; }$ , is the negative log-likelihood of the original counts under the predicted NB distribution, computed for the masked genes across all cells in the cell set ??: 

$$
\mathcal {L} _ {\text { recon }} = - \frac {1}{K | \mathcal {M} |} \sum_ {k = 1} ^ {K} \sum_ {g \in \mathcal {M}} \log P (x _ {g} ^ {(k)} | l ^ {(k)}, \rho_ {g} ^ {(k)}, \theta_ {g} ^ {(k)}). \tag {4.5}
$$

Optimizing solely the reconstruction loss leads to memorization and a reduction in performance. To enforce meaningful pre-training, we incorporate a regularization term, ${ \mathcal { L } } _ { S W ; }$ , defined as the Sliced Wasserstein distance between the empirical distribution of the final flattened cell states $\{ \bar { \mathbf { E } } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ and a batch-centered, multivariate Gaussian prior $\begin{array} { r } { N ( \frac { 1 } { K } \sum _ { k = 1 } ^ { K } { \bar { \mathbf { E } } ^ { ( k ) } } , \mathbf { I } _ { n d } ) } \end{array}$ . This term enforces the embedding to be decomposed into a centralized Gaussian distribution and a cell-set-specific constant vector, which regularizes embedding distribution and enforces linear identifiability of latent factors (Dong et al., 2024). The linear identifiability result follows directly from the theoretical framework established in Khemakhem et al. (2020). The loss term is calculated within a random cell set subset, with subset size uniformly sampled from [32, 33, · · · , 128]. The hyperparameter ??SW (default 0.01) balances the two loss components. 

# 4.2. Post-training

The pre-trained Stack model is post-trained for in-context prompting tasks through a supervised alignment procedure. The objective is to empower the model to generate novel cell populations, combining cell type information from a set of query cells, and the biological context provided by a separate set of prompt cells. Our approach comprises four key components: (i) a novel self-distillation procedure, (ii) a training input construction strategy, (iii) architectural modifications, and (iv) a composite loss function that balances generalization with knowledge retention. 

# 4.2.1. Self-distillation

We introduce a self-distillation procedure during post-training. A frozen teacher model, operating without input data masking, computes embeddings and gene expression parameters from cell sets containing individual biological samples; these outputs are then used to calculate the training objectives. The teacher model weights are updated via exponential moving average (EMA) of the student model which is actively post-trained. This approach follows the self-distillation framework established in self-supervised vision models (Grill et al., 2020; Oquab et al., 2023; Zhou et al., 2021). 

# 4.2.2. Post-training input construction

Different from pre-training, the post-training process involves two biological samples with matched cell type annotations. We begin with a cell set of ?? cells $\mathbf { \bar { X } } = \{ \mathbf { x } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ 1 drawn from the prompt sample. Cells are ordered such that cells of the same type appear consecutively, with cell type order randomized for each sample. This cell set is partitioned into three components: 

1. Prompt condition cells $( \mathbf { X } _ { \mathrm { p r o m p t } } ^ { \mathrm { f i x e d } } )$ : The first 25% of cells, used unchanged as conditioning context. 

2. Prompt context cells $( \hat { \mathbf { X } } _ { \mathrm { p r o m p t } } ^ { \mathrm { k e p t } } ) \colon K _ { \mathrm { k e p t } }$ cells whose expression profiles are sampled from the means andack 

3. Target cell positions: The remaining $K _ { \mathrm { q u e r y } }$ positions, where $K _ { \mathrm { k e p t } } + K _ { \mathrm { q u e r y } } = 0 . 7 5 K$ and the ratio $K _ { \mathrm { k e p t } } / ( K _ { \mathrm { k e p t } } + K _ { \mathrm { q u e r y } } ) \sim \mathcal { U } ( 0 , 1 )$ . 

The target cell positions are filled with query cells $\mathbf { ( X _ { q u e r y } ) }$ drawn from a different biological context (e.g., a different donor or perturbation condition). Each query cell is matched by cell type to the target cell it replaces. This yields the final input: 

$$
\mathbf {X} _ {\text {in}} = \left[ \mathbf {X} _ {\text {prompt}} ^ {\text {fixed}}, \hat {\mathbf {X}} _ {\text {prompt}} ^ {\text {kept}}, \mathbf {X} _ {\text {query}} \right]. \tag {4.6}
$$

The model’s objective is to predict the gene expression distribution of target cells, conditioning on the prompt cells as reference. To ensure the model learns meaningful biological transitions, each replaced cell must have the same cellular identity (e.g., cell type or cell line, specified by user) as the original cell it replaces. To address imbalanced cell type distributions, we apply a balancing procedure to each training sample: overrepresented cell types are downsampled to the average count per type, then the resulting pool is upsampled with replacement to restore the original cell set size. For replicated cells, only the first instance contributes to the distributional loss terms which will be detailed in Section 4.2.4. 


Table 1 | Components of the post-training input cell set ${ \bf X } _ { \mathrm { i n } }$ .


<table><tr><td>Component</td><td>Notation</td><td>Size</td><td>Description</td></tr><tr><td>Prompt condition</td><td><eq>X_{prompt}^{fixed}</eq></td><td>0.25K</td><td>Original cells from prompt sample</td></tr><tr><td>Prompt context</td><td><eq>\hat{X}_{prompt}^{kept}</eq></td><td><eq>K_{kept}</eq></td><td>Sampled from teacher-predicted distributions</td></tr><tr><td>Query</td><td><eq>X_{query}</eq></td><td><eq>K_{query}</eq></td><td>Cells from a different biological context</td></tr><tr><td colspan="4">Note: <eq>K_{kept} + K_{query} = 0.75K</eq>; ratio <eq>K_{kept}/(K_{kept} + K_{query}) \sim \mathcal{U}(0,1)</eq>.</td></tr></table>

# 4.2.3. Architectural modifications

We introduce two learnable modules to the model: a query position embedding, $\mathbf { P } _ { \mathrm { q u e r y } } \in \mathbb { R } ^ { n d }$ and an MLP binary classifier $f _ { \mathrm { C L S } } : \mathbb { R } ^ { 2 n d } \to \mathbb { R }$ . The position embedding $\mathbf { P } _ { \mathrm { q u e r y } }$ is added to the token representations of query cells at the first embedding layer. The MLP binary classifier $f _ { \mathrm { C L S } } : \mathbb { R } ^ { 2 n d } \to$ ℝ that takes the concatenation of mean prompt condition embedding and prompt context/query single-cell embedding as input, and predicts whether the cell comes from prompt (0) or query (1). The classifier receives detached embeddings as input, therefore its optimization is independent from other model weights. We register gradient hooks on these newly introduced modules to apply a 10× gradient scaling, which effectively increases their learning rates relative to the pre-trained parameters. Additionally, a causal attention mask is applied within all transformer layers to prevent prompt condition cells from attending to prompt context or query cells, ensuring that information flows strictly from prompt condition cells to the remaining cells. 

# 4.2.4. Post-training objective

Similar to the pre-training setup, the model receives a masked version of the input cell set ${ \bf X } _ { \mathrm { i n } }$ with rectangular gene masks, with masking ratio sampled from U (0.1, 0.3). The post-training objective is designed to simultaneously predict unseen target cells and maintain information learned during pre-training: 

$$
\mathcal {L} _ {\mathrm{FT}} = \mathcal {L} _ {\text {dist}} + \lambda_ {\text {recon}} \mathcal {L} _ {\text {recon}} + \lambda_ {\mathrm{SW}} \mathcal {L} _ {\mathrm{SW}} + \lambda_ {\mathrm{CLS}} \mathcal {L} _ {\mathrm{CLS}}; \quad \mathcal {L} _ {\text {dist}} = 0. 5 \times (\mathcal {L} _ {\text {gene}} + \mathcal {L} _ {\text {embed}}). \tag {4.7}
$$

The components are defined as follows: 

• Embedding Alignment Loss $( \mathcal { L } _ { \mathrm { e m b e d } } )$ : The term is defined as the energy distance between Stack embeddings of query cells and target cells. The embedding for the target cells is extracted from the teacher model and is detached for loss calculation. The student model to be fine-tuned calculates the embedding for query cells from input data cell set ${ \bf X } _ { \mathrm { i n } }$ . 

• Expression Alignment Loss $\scriptstyle ( { \mathcal { L } } _ { \mathrm { g e n e } } )$ : The term is defined as the energy distance between the predicted and true distributions of log normalized target gene expression. To make the optimization on NB distribution parameters tractable, predictions are generated using a reparameterizable zero-inflated normal distribution sampler that matches first two moments of the log-normalized NB distribution. To address over-smoothing, we estimate a shared over-dispersion parameter for query cells using the median detached over-dispersion computed from prompt cells. The loss is computed on first 1,000 highly variable genes per mini-batch (identified via Pearson residuals (Lause et al., 2021)), and is stratified by cell type like Lembed. $\mathcal { L } _ { \mathrm { e m b e d } }$ 

• Distributional Alignment Loss $( \mathcal { L } _ { \mathrm { d i s t } } )$ : This is the primary alignment objective, which is defined as the average of embedding alignment loss $\mathcal { L } _ { \mathrm { e m b e d } }$ and gene expression alignment loss $\mathcal { L } _ { \mathrm { g e n e } }$ . 

• Reconstruction Loss $( { \mathcal { L } } _ { \mathrm { r e c o n } } ) \colon$ To retain the model’s mask reconstruction capabilities, we apply a standard masked gene reconstruction loss, as in pre-training, to the prompt cells. This serves as an auxiliary task that regularizes the model and enforce it to maintain pre-training objective. We use $\lambda _ { \mathrm { { r e c o n } } } = 1$ . 

• Latent Regularization $( \mathcal { L } \boldsymbol { s } \mathbf { w } )$ : The Sliced Wasserstein distance objective from pre-training is retained and applied to the embeddings of all cells in the batch. We use $\lambda _ { S W } = 0 . 0 1$ . 

• Classification Loss $( \mathcal { L } _ { \mathbf { C L S } } ) \mathrm { : }$ : The classification loss is defined as the binary cross entropy loss (BCE) of the MLP classifier in classifying query from prompt context cells. We use $\lambda _ { \mathrm { C L S } } = 1$ . 

# 4.2.5. Generative procedure

The post-training setup closely resembles a conditional mask diffusion model, where prompt cells represent unmasked tokens and query cell represent masked tokens [mask] that the model must predict. Two key differences distinguish Stack’s generative inference: 

1. The gene expression profile of query cells is inputted to the model in order to encode cell type information. Combined with $\mathbf { P } _ { \mathrm { q u e r y } } .$ , it serves a similar role to positional encodings of [mask] tokens in language models. 

2. Masked language models derive token-level confidence directly from softmax probabilities over the vocabulary. Since our output space is continuous, we instead train a separate classifier to estimate prediction confidence, which guides the selective unmasking procedure during generation. 

At each generative step, the model receives mini-batches containing concatenated prompt condition cells, prompt context cells and query cells. Here, prompt context cells correspond to original cell expression profiles. The ratio of prompt condition cells remains constant at 25%, while the ratio of prompt context cells increases linearly from 0.2 to 0.4 throughout the generative process. A boolean array is_mask is maintained to indicate whether each query position remains to be predicted, and is initialized with all True for query cells. Each step consists of a prediction-and-update cycle guided by a linear masking schedule $( 1 - t / T )$ : 

• Prediction: The model performs a forward pass, generating a complete expression profile for all query cells. 

• Confidence Scoring: The classifier module assesses each predicted query cell, and outputs a logit vector. Positive values indicate the cell is more like a query cell, negative values indicate the cell is more like a prompt cell. 

• Selective Unmasking: Based on the masking schedule, a fraction of query cells with is_mask=True are selected to be replaced with model prediction. The cells with the lowest logit values are chosen for the replacement. 

• State Update: The replaced cells’ is_mask values are set to False. Next, all query cells with logit value $> 0 \ ( \mathrm { i . e . }$ , those classified as query cells) are (re-)set to is_mask=True. 

The iterative process concludes when the masking rate in the schedule reaches zero. The final output from this step constitutes the complete, generated expression matrix for the initial set of query cells. The final logit vector is also a part of the output for quantifying generation quality and interpretation. 

# 4.3. Model implementation

The Stack model is implemented in PyTorch and trained using the PyTorch Lightning framework. The training process consists of two stages: self-supervised pre-training and supervised post-training for in-context prompting. 

# 4.3.1. Base model

The Stack architecture consists of $N _ { L } \in \{ 6 , 9 \}$ tabular transformer layers. Each cell is tokenized into ?? = 100 tokens, each with dimension ?? ∈ {8, 16, 32}, yielding a total per-cell embedding dimension ???? ∈ {800, 1600, 3200}. The input to the transformer is a cell set of ?? = 256 cells tokenized. The feed-forward network within each layer uses a GELU activation function. The decoder is implemented as a 2-layer MLP with GELU activation. No dropout was applied during pre-training. Configurations of different model settings are detailed in the scaling study section. 

# 4.3.2. Pre-training data

The pre-training data was sourced from the full human scBaseCount, the scBaseCount subset, or the human CELLxGENE. For scBaseCount, we filtered cells with 300–7,000 detected genes and at least 700 UMIs. No filtering was performed for CELLxGENE datasets. For model training, a unified gene list was created by computing the union of the top 1,000 highly variable genes (HVGs) from each scBaseCount data file, capped at a maximum of 15,012 total genes. HVGs were identified using analytic Pearson residuals (Lause et al., 2021). The dataloader generates samples by creating non-overlapping, contiguous chunks of ?? = 256 cells from each file. Cells in chunks shorter than ?? are dropped. Pre-training data metrics are summarized in Table 2. 


Table ${ \mathbf { 2 } } \mid { \mathrm { A n } }$ overview of training datasets for Stack models in this study.


<table><tr><td>Training sets</td><td>#Training Datasets</td><td>#Training Cells</td></tr><tr><td>CellXGene (§)</td><td>905</td><td>73.7M</td></tr><tr><td>scBaseCount subset (†)</td><td>9004</td><td>60.2M</td></tr><tr><td>scBaseCount full set (‡)</td><td>19978</td><td>148.8M</td></tr></table>

# 4.3.3. Pre-training setup

The Stack base/large model was pre-trained for 10 epochs using the AdamW optimizer with a peak learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of $3 \times 1 0 ^ { - 3 }$ . For Stack XLarge and Huge models, the peak learning rate was tuned down to $3 \times 1 0 ^ { - 5 }$ to stabilize training. A cosine annealing learning rate schedule was used with a linear warmup over the first epoch. The Sliced Wasserstein regularization weight ??SW was set to 0.01. All experiments were conducted on a single NVIDIA H100 GPU (80GB HBM) with 320GB system RAM, using bf16 mixed precision with a batch size of 32 and 4 training data-loading workers. We benchmarked data-loading and pre-training speed against a State Embedding model under equivalent settings. 

# 4.3.4. Post-training data

The dataset for supervised alignment was curated from multiple public sources, including the Parse 10M PBMC data (Parse Biosciences, 2023), and a programmatically selected subset of the CELLxGENE database (Program et al., 2025). To ensure suitability for learning donor-specific effects, a CELLxGENE subset was filtered to include only large-scale datasets (> 50, 000 cells) with at least five unique donors. As the default "cell_type" column in CELLxGENE is found to be suboptimal, we implemented an automatic procedure to identify optimal cell type annotation column for each dataset, employing a heuristic algorithm that prioritizes author-provided, intermediate-granularity labels $( \mathrm { e . g . }$ , ‘author_cell_type’, ‘ann_coarse’) over standardized ontologies or overly detailed subtypes. The dataset selection procedure resulted in 45 million cells from 189 datasets. The posttraining dataloader first splits the curated datasets into training, validation, and test sets based on donor or sample ID. Each training sample consists of a cell set of $K = 5 1 2$ cells, which is further partitioned into $K _ { \mathrm { k e p t } } = 1 2 8$ prompt condition cells and $K _ { \mathrm { q u e r y } } = 3 8 4$ prompt context/target cells. 

# 4.3.5. Post-training setup

The model was fine-tuned for 8 epochs, starting from the pre-trained weights from the Stack (large) model trained on the full human scBaseCount. We used the AdamW optimizer with a peak learning rate of $2 \times 1 0 ^ { - 5 }$ and a weight decay of $3 \times 1 0 ^ { - 3 }$ . A cosine annealing learning rate schedule with a 1-epoch linear warmup and min learning rate $5 \times 1 0 ^ { - 6 }$ was applied. For teacher model updates, we employed an exponential moving average (EMA) with a decay rate of 0.95, applied every 500 optimization steps. The training was configured with a batch size of 8 and 4 steps of gradient accumulation, resulting in an effective batch size of 32. Finetuning experiments were conducted on a single NVIDIA H100 GPU (80GB HBM) with 400GB system RAM, using bf16 mixed precision. 

# 4.4. Ablation and scaling studies

We performed comprehensive ablation and scaling studies to evaluate the performance of Stack across settings and scales. An overview of tested models is shown below. All models use hidden dimension $d = 1 0 0$ and attention heads $N _ { H _ { C } } = N _ { H _ { G } } = 8 _ { ; }$ , except for XLarge and Huge models which use $N _ { H _ { C } } = 2 0$ . 


Table 3 | An overview of model settings tested in the scaling study.


<table><tr><td>Model Variant</td><td>Layers <eq>N_L</eq></td><td>Total Embed Dim nd</td><td>Params</td><td>Non-emb Params</td></tr><tr><td>STACK (Base−) †</td><td>3</td><td>800</td><td>69.1M</td><td>57.0M</td></tr><tr><td>STACK (Base) §, †, ‡</td><td>6</td><td>800</td><td>76.7M</td><td>64.7M</td></tr><tr><td>STACK (Base+) †</td><td>9</td><td>800</td><td>84.4M</td><td>72.4M</td></tr><tr><td>STACK (Medium) †</td><td>6</td><td>1600</td><td>186M</td><td>163M</td></tr><tr><td>STACK (Large) †, ‡</td><td>9</td><td>1600</td><td>217M</td><td>193M</td></tr><tr><td>STACK (XLarge) ‡</td><td>6</td><td>3200</td><td>506M</td><td>459M</td></tr><tr><td>STACK (Huge) ‡</td><td>9</td><td>3200</td><td>629M</td><td>582M</td></tr></table>

We also evaluated two ablations of Stack (Base) †: (i) without latent regularization, and (ii) without both latent regularization and inter-cellular attention. 

# 4.5. Stack embedding evaluations

# 4.5.1. Probing evaluation

To assess the biological information encoded in the learned cell representations, we implemented a multitiered probing framework. All probing experiments employed a group-based splitting strategy, where datasets were partitioned by donor ID for observational data and PBMC perturbation data, or by 50% sample split (grouped by library label) for cell-line perturbational data. This ensures that models are evaluated on cells from entirely unseen donors (or library splits for cell-line datasets). To address class imbalance in the test set, we capped the maximum contribution per donor at 2,000 cells for observational data, 25,000 cells for Parse, and 100,000 cells per sample split for other perturbation datasets. Two main probing approaches were employed to evaluate different aspects of the learned representations: 

• Linear probing: Logistic regression for classification tasks and ridge regression for continuous variables, trained with 5-fold cross-validation on 80% of donors. This setting trains separate models for each cell type, enabling assessment of cell-type-specific information encoding. All analyses were performed on the top 20 most abundant cell types if the total cell type number exceeds 20 (or 5 per tissue in Tabula Sapiens). 

• MLP probe: For capturing non-linear relationships, we implemented an MLP consisting of an input layer normalization, a hidden layer with 128 units and ReLU activation, dropout of 0.2, and a task-specific output layer. The model was optimized using AdamW with learning rate $1 \times 1 0 ^ { - 3 }$ for up to 80 epochs with early stopping (patience of 12 epochs). L2 regularization strength was selected via grid search over 

$\{ 0 , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ based on validation loss, using a 70/15/15 train/validation/test split by donor. The model is trained on 5 most abundant cell types together for each dataset. 

Performance was evaluated at the single-cell level. Since the test set caps each donor’s contribution, the calculated cell-level accuracy metrics equal donor-level average metrics when all donors exceed the 2,000- cell threshold, and smoothly approximate it when some donors have fewer cells. For classification tasks, we report balanced accuracy. For regression tasks, we report Pearson correlation coefficients. All evaluations were performed with a fixed random seed of 42. 

# 4.5.2. Batch integration evaluation

We performed batch integration evaluations using the scib-metrics package (v0.5.5). To remove extremely rare cell populations that lead to benchmarking errors, the analysis was conducted on a subset containing the top 20 most abundant cell types, or the top 5 most abundant cell classes for Tabula Sapiens. If the resulting cell count exceeded 100,000, the subset was randomly downsampled to this size. The evaluation was structured around two primary objectives, each assessed by a specific suite of metrics. 

1. Batch Correction: To quantify the removal of technical batch effects, we measured the k-nearest neighbor Batch Effect Test (kBET), graph connectivity, principal component regression (PCR) score, integration Local Inverse Simpson’s Index (iLISI), and batch-removal-adapted silhouette (BRAS) (Rautenstrauch and Ohler, 2025). 

2. Biological Conservation: To assess preservation of biological signal, we measured Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI) of Leiden clusters against ground-truth cell type labels, silhouette score for cell type labels, and the cell type Local Inverse Simpson’s Index (cLISI). 

These scores are aggregated to compute the total score, following the scib-metrics default (Luecken et al., 2022). 

# 4.5.3. Baseline models

The Stack embedding was benchmarked against several methods: 

1. PC HVG: We performed library normalization, log1p transformation, and Scanpy default highly variable gene selection to identify the top 2,000 highly variable genes. These genes were then reduced to 800/50 principal components. The former setting, matching the dimensionality of Stack (base), excels at probing tasks, while the latter is better for integration evaluations. 

2. scGPT (v0.2.4): We used the recommended whole-human scGPT checkpoint to generate cell embeddings. Input data was preprocessed with library size normalization and log1p transformation, with a batch size of 256 for inference. 

3. UCE (commit 8227a65): We employed the 33-layer model checkpoint in (Rosen et al., 2023) with batch size 20 to extract embeddings. The model operates directly on raw count data without normalization. As UCE by default can skip an extremely small amount of cells due to preprocessing, we aligned the output embeddings back to the original cell indices, filling missing cells with zeros. 

4. State (SE) (state v0.9.27): We used the SE-600M checkpoint on huggingface and the state emb transform command to infer embeddings on raw count data. Data loader throughput was measured using: uv run state emb fit model.batch_size=32 optimizer.gradient_accumulation_steps=256 dataset.num_train_workers=4. 

5. TranscriptFormer (v0.6.1): We used the TF-sapiens checkpoint from (Pearce et al., 2025) due to its advantage on human scRNA-seq data evaluations. We used the provided CLI to infer embeddings from raw count data. 

6. scVI fine-tuned (scvi-tools v1.3.1): Pre-trained scVI model initially trained on the scBaseCount subset for 10 epochs, with 2 hidden layers of 2000 hidden dimensions and 800 latent dimensions, or 256 hidden dimensions and 50 latent dimensions. The model was subsequently fine-tuned on each target dataset for 20 additional epochs with learning rate $5 \times 1 0 ^ { - 4 }$ . The fine-tuning employed a streaming mini-batch strategy: samples of 256 cells were drawn from individual files and concatenated to form training batches of 4096 cells. Each cell was tagged with a batch ID derived from its source file. The strategy greatly accelerates scVI training on large single-cell data collections. 

7. scVI from scratch (scvi-tools v1.3.1): Models were configured with 2 hidden layers of 2000/128 hidden dimensions and 800/30 latent dimensions, and trained for 50 epochs using Adam optimizer with learning rate $1 \times 1 0 ^ { - 3 }$ and weight decay $1 \times 1 0 ^ { - 4 }$ . 

# 4.5.4. Evaluation datasets

The five observational studies for probing and integration evaluations are downloaded from the CELLxGENE portal (Program et al., 2025) (Kidney, BCL, SEAS-AD MTG, LUCA, and Tabula Sapiens). Perturbation datasets are downloaded from their official websites (Tahoe, Parse, X-Atlas:Orion (Xaira), and the OpenProblems competition). We balanced the tissue composition of Tabula Sapiens, downsampling cell in each tissue to 20,000 if the cell number exceeds the threshold. 


Table 4 | Overview of datasets used in this study. ∗ indicates subsampling from the original dataset.


<table><tr><td>Dataset Category</td><td>Dataset Name</td><td># Donors</td><td># Cell Types</td><td>Conditions</td></tr><tr><td rowspan="5">Observational Atlases</td><td>Kidney atlas</td><td>77</td><td>43</td><td>Disease: 14 (AKI) / 37 (CKD) / 26 (Healthy)Hypertension: 41 (Yes) / 36 (No)Diabetes history: 38 (Yes) / 38 (No)</td></tr><tr><td>Brain atlas (SEAS MTG)</td><td>38</td><td>65</td><td>Microinfarct pathology: 34 (0–3) / 2 (4–6) / 2 (7–10)ADNC: 4 (Not AD) / 7 (Low) / 9 (Intermediate) / 18 (High)Braak stage: 2 (0) / 2 (II) / 4 (III) / 8 (IV) / 11 (V) / 11 (VI)Thal phase: 4 (0) / 3 (1) / 4 (2) / 7 (3) / 10 (4) / 10 (5)CERAD score: 9 (Absent) / 5 (Sparse) / 7 (Moderate) / 17 (Frequent)APOE4 status: 23 (N) / 15 (Y)</td></tr><tr><td>Lymph node atlas (BCL)</td><td>223</td><td>21</td><td>Disease: 208 (BCL) / 15 (Healthy)LymphoMAP: 84 (FMAC) / 76 (LN) / 64 (TEX)</td></tr><tr><td>Lung atlas (LUCA)</td><td>160</td><td>39</td><td>Disease: 18 (COPD) / 75 (LUAD) / 17 (LUSC) / 2 (NSCLC) / 48 (Healthy)UICC stage: 45 (I) / 17 (II) / 7 (III) / 24 (IV) / 66 (non-cancer)Ever smoker: 85 (Yes) / 75 (No)</td></tr><tr><td>Tabula Sapiens</td><td>24</td><td>31</td><td>25 tissues with &gt;1 donors</td></tr><tr><td rowspan="4">Perturbational Datasets</td><td>OpenProblems-PBMC</td><td>3</td><td>6</td><td>147 conditions</td></tr><tr><td>Tahoe-100M</td><td>-</td><td>40*</td><td>250* perturbations from plates 1-3</td></tr><tr><td>Parse-PBMC</td><td>12</td><td>17</td><td>90 perturbations</td></tr><tr><td>X-Atlas:Orion (Xaira)</td><td>-</td><td>2</td><td>Top 50* perturbations per cell line</td></tr></table>

# 4.6. Cell prompting task evaluation

The prompting evaluation framework assesses how well a model can generate desired expression profiles given prompt and query cells through in-context learning (ICL). We evaluated four ICL settings: 


Table 5 | Overview of in-context cell prompting tasks for Stack.


<table><tr><td>Task Category</td><td>Task Name</td><td>Prompt</td><td>Query</td><td>Same Dataset?</td></tr><tr><td rowspan="2">Perturbational ICL</td><td>1. Perturbation effect prediction for novel cell types</td><td>Perturbed cells (random types)</td><td>Control cells (non-overlapping types)</td><td>Yes</td></tr><tr><td>2. Perturbation effect prediction for novel samples</td><td>Perturbed T cells (Donor A)</td><td>Unperturbed T cells (Donor B)</td><td>No</td></tr><tr><td>Observational ICL</td><td>3. Hold-out cell type prediction</td><td>Selected cell types (Donor A)</td><td>Non-overlapping types (Donor B)</td><td>Yes</td></tr><tr><td>Hybrid ICL</td><td>4. Cross-dataset cell type generation</td><td>Selected cell types (Donor/Pert A)</td><td>Non-overlapping types (Donor B)</td><td>No</td></tr></table>

# 4.6.1. Evaluation data construction

In settings with cell type hold-outs (1, 3, 4), a set of broad cell classes present in one donor/condition was randomly sampled and held out (ratio 0.75) as target cells to be predicted. The remaining non-held-out cell types from this donor formed the prompt data, while held-out cell types from another donor/condition were used as query cells. In response prediction across samples (settings 2), all data were subsetted to a single cell type (T cells). In perturbational data (settings 1–2), control conditions from the same prompt donor are typically available. Therefore, we additionally utilized these cells as auxiliary prompts to the model and used the output as a “synthetic control” for cell-eval benchmarking. This synthetic control approach was applied to Stack and other baselines when appropriate. For observational prompting tasks, oracle baselines additionally used non-held-out cell types in query conditions as auxiliary data, which were not available to Stack. We equalized the number of query, auxiliary, and target cells by downsampling to the minimum available cell count among all three groups. To account for rare cell types in settings 1 and 3, we upsampled each cell type in the query data to a minimum of 2000 and 1000 cells, respectively. 

# 4.6.2. Baseline models

The prompting performance of Stack was benchmarked against several methods: 

1. Original Query: The original query cells are used for prediction, representing a zero-change scenario where no information from the prompt is used. 

2. Nearest Cell Type in Prompt: This baseline computes pseudobulk expression profiles for all cell types in the prompt data. For each query cell, it identifies the most similar cell type in the prompt based on pseudobulk Pearson correlation, then assigns an expression profile from a randomly sampled cell of that matched cell type without replacement. Once all cells from a cell type are exhausted, the second-closest cell type is used, and so on. 

3. Same Cell Type in Prompt: This baseline assigns expression profiles by sampling from cells of the same cell type in the prompt without replacement when possible. 

4. PerturbMean/DonorMean: This method computes the average difference in expression profiles (per cell type) between the prompt and query contexts. This difference vector is calculated on non-held-out cell types between prompt and auxiliary data, then added to the expression profiles of the query cells. The synthetic control is not applicable for this baseline, as it would be equivalent to the query data. 

5. scVI: A 2-layer scVI model (scvi-tools v1.3.1) with 30 latent dimensions and 200 hidden dimensions was trained on combined prompt and auxiliary samples, using dataset origin as the batch key, for 50 epochs. The model was then used to project query cells into the latent space and sample their expression profiles, artificially specifying “prompt” as the batch label. 

6. State: State models (v0.9.27) were trained using the ST+SE setting (Adduri et al., 2025), where the model predicts cell embeddings and simultaneously decodes them back to gene expression space. Models were trained with Maximum Mean Discrepancy (MMD) loss on prompt, query, and auxiliary data, using the union of highly variable genes in cell-eval benchmarks across random seeds. Only control cells from the query set were used for model training. All models used a hidden dimension of 328 and cell set length of 32. Models were for 60,000 steps (batch size 8, learning rate $3 \times 1 0 ^ { - 4 } )$ . Random basal mapping was employed with gene-space outputs. The best checkpoints were selected via validation loss on held out cells. 

# 4.6.3. Evaluation metrics

We evaluated the generated expression profiles against ground truth data using pseudobulk correlation and DE metrics. Both categories of metrics are implemented using cell-eval v0.6.6 (Adduri et al., 2025) with default parameters, employing Wilcoxon rank-sum tests for DE detection and Benjamini-Hochberg correction for multiple testing. All metrics were computed on top 2000 log-normalized highly variable genes, identified from the concatenation of target and query data. For setting 4, we additionally evaluated the integration performance of ground truth and predicted gene expression profiles using scIB, with the same configuration as the earlier benchmarking. 

• Pseudobulk correlation. We measured how well methods capture pseudo-bulk level perturbation effects by two metrics: 

– Pearson Delta: Pearson correlation between predicted and observed expression changes. For each perturbation $t ,$ we calculate the expression delta as $\Delta _ { t } = \left| p _ { t } - p _ { \mathrm { c t r l } } \right|$ for both predicted $( \bar { \Delta } _ { t } )$ and ground truth $\left( \Delta _ { t } \right)$ pseudobulks, then compute: Pearson- $\begin{array} { r } { \cdot \Delta = \mathrm { c o r r } ( \bar { \Delta } _ { t } , \Delta _ { t } ) } \end{array}$ 

– DE Spearman LFC: the Spearman rank correlation between predicted and observed log fold changes, calculated within the set of significantly differentially expressed (DE) genes in the ground truth. 

– DE direction match: This metric measures whether the predicted direction of gene expression change (i.e., up- or down-regulation) matches the ground truth. It is calculated only on the set of genes that are significantly differentially expressed (DE) in both the predicted and true data. The score is the fraction of these shared DE genes for which the direction of change matches. This metric replaces Pearson Delta in observational prompting tasks. Pearson Delta performs comparably to DE Spearman LFC when many genes are identified as DEGs, while also showing greater vulnerability to batch effects. 

• Differential expression accuracy. Finally, we evaluated whether the prediction captures differential expression between ground truth data and the input query condition. 

– PR-AUC: Area under precision-recall curve using binary DE labels and $- \log _ { 1 0 } ( p \mathrm { - v a l u e s } )$ as scores, using sklearn average precision score implementation. 

– Spearman effect size: To compare the relative effect sizes of perturbations, we calculate Spearman correlation coefficients on the number of differentially expressed genes (adjusted p-value < 0.05) between predicted and ground truth. This assesses whether models accurately capture relative effect sizes across different conditions. We averaged predictions across random seeds within each cell type and perturbation/donor, yielding a Spearman correlation computed across perturbations/donors per cell type. 

– DE Overlap Accuracy: Computes the overlap between the top-?? genes from the true DE abs-logfold-change ranking and the top-?? genes from the predicted DE ranking, calculated as |top-?? true∩ top-?? predicted|/??, where ?? is the total number of true DE genes. 

– DE Precision-at-??: Computes the overlap between the top-?? genes from the true DE ranking and the top-?? genes from the predicted DE ranking, calculated as |top-?? true ∩ top-?? predicted|/??, where ?? is the total number of predicted DE genes. This metric assesses the fidelity of predicted DE gene lists and replaces Spearman effect size correlation in Setting 2, where the limited number of perturbations with similar effect sizes makes rank-based correlation unreliable. 

– Jaccard similarity: For each perturbation, we compute the Jaccard index between the set of predicted DE genes and ground truth DE genes, defined as the size of the intersection divided by the size of the union of the two sets. 

# 4.6.4. Evaluation datasets

For evaluations, we used 1. OpenProblems drug perturbation (Luecken et al., 2025), 2. Cytokine stimulation (Dong et al., 2023), 3. Immune aging (Wells et al., 2025), 4. Tabula Sapiens (Consortium* et al., 2022), 5. Kidney atlas (De Boer et al., 2021), 6. Lymph node BCL (Li et al., 2025), 7. Liver atlas (Edgar et al., 2025), 8. Parse cytokine perturbation dataset (Parse Biosciences, 2023). Observational atlases (3-7) were downloaded from the CELLxGENE portal, and the cytokine stimulation data was downloaded from Dryad (Dong et al., 2023). The cytokine stimulation data (Dong et al., 2023) comprise 3 donors, but donor 1 has very limited cell numbers. Therefore, we subsetted the dataset to include only donors 2 and 3 (relabeled as donors a and b) and only included cells from the acute condition (2 days). 

For setting 1, the OpenProblems dataset used prompts and queries sampled from the same donor (3 donors total) within each test experiment. For settings 2, prompts used T cells from one Parse donor under cytokine conditions (IFN-??, IFN-??, IL-6, TNF-??), while queries used control T cells from donors a or b in (Dong et al., 2023). Ground truth evaluation used corresponding individual and combination conditions (IFN-??, IFN-??, IL-6, TNF-??, IFN-??+IFN-??, IFN-??+IL-6, IFN-??+TNF-??) from (Dong et al., 2023). For combinatorial perturbation evaluations, we computed weighted sums of single-perturbation Stack predictions. Due to dosage differences between the Parse dataset and Dong et al. (2023), we performed a grid search over weights [0.1, 0.3, 0.5, 0.7, 0.9] to determine the optimal weighted average of normalized gene expression conditions in the original prompt T cells. The optimal weight was selected based on Pearson Delta and subsequently used to generate Stack predictions for combinatorial perturbations. For setting 3, prompts and queries comprised nonoverlapping cell types sampled from different donors in each dataset. For setting 4, prompts comprised: Parse control cells (PBS condition) with one donor sampled per prompt; Parse donor 1 cells with one perturbation sampled per prompt; immune aging dataset cells with one donor per prompt; OpenProblems cells with one perturbation per prompt. All queries in this setting used immune cells from Tabula Sapiens. In settings 3 and 4, we performed cell class-level hold-out for each evaluated dataset, relabeling cell annotations from original author annotations to align with the broad cell classes defined in Tabula Sapiens for corresponding tissues. 

Holding out at the cell class level minimizes potential information leakage across similar cell types. 

# 4.7. Whole-organism perturbational atlas Perturb Sapiens

In the analysis, we used each condition from donor 2 in OpenProblems drug perturbation data and donor 1 in Parse cytokine perturbation data as the prompt, and the tissue balanced version of Tabula Sapiens as the query. We used post-trained Stack (T=5) to perform in-context generation. We removed cells with classifier logits above 2.5. 

We generated log fold change heatmaps for individual cytokine treatments. For each cell type, up to 10,000 cells per condition were subsampled, normalized, and log-transformed. Genes were subsetted to 4,000 highly variable genes identified via Pearson residual in the example ADSF Perturb Sapiens dataset (Lause et al., 2021). Differential expression analysis was performed using the Wilcoxon rank-sum test with Benjamini-Hochberg correction (DEG: FDR < 0.05, |log FC| > 0.25/0.5 for Parse/OpenProblems respectively). To compare Perturb Sapiens performance against biological replicates across drug or cytokine perturbations, the same differential expression procedure was applied, with each condition subsampled to a maximum of 15,000 cells. Cell-eval evaluation metrics were constructed as described above, with one modification: significant genes were defined using an explicit minimum log fold change threshold, consistent with the DEG procedure. 

For evaluation on non-immune cells, we downloaded four epithelial in vitro cytokine perturbation datasets from the GEO database (Koh et al., 2023; Swindell et al., 2018; Lee et al., 2022; Saito et al., 2021). All evaluations were restricted to the top 4,000 highly variable genes identified from the example ADSF Perturb Sapiens. For the first three datasets, log fold changes, ??-values, and adjusted ??-values were computed using PyDESeq2 (v0.5.2) on either pseudobulk (Koh et al., 2023) or bulk RNA-seq data (Swindell et al., 2018; Lee et al., 2022). For (Saito et al., 2021), due to the normalized TPM format of the deposited data, we performed a donor-wise paired ??-test to calculate log fold changes, ??-values, and adjusted ??-values using Benjamini–Hochberg correction. Differential expression analysis on Perturb Sapiens was performed using Scanpy’s Wilcoxon rank-sum test implementation. Airway and keratinocyte epithelial populations in Perturb Sapiens were constructed by selecting cell classes annotated as “epithelial” from their respective tissues (lung, trachea for airway; skin and tongue for keratinocytes). To ensure fair comparison across Perturb Sapiens cell types, each cell type population was capped at a maximum of 15,000 cells. We applied an LFC threshold of 0.5 for in vitro single-cell data evaluations and 0.25 for bulk data evaluations. Finally, for each drug/cytokine perturbation and its matched control, we identified tissue–cell type combinations with sufficient cell counts (≥ 1000 cells). We then applied the same differential expression analysis pipeline described above to compute log-fold changes and adjusted ??-values. In Figs. S17 and S18, perturbation effects are defined as log fold changes, with non-significant values (adjusted p > 0.05) set to zero. 

Data availability. Documentation on accessing scBaseCount can be found at https://github.com/ArcInstit ute/arc-virtual-cell-atlas. The generated Perturb Sapiens data is deposited on Huggingface: https://huggingf ace.co/datasets/arcinstitute/Perturb-Sapiens. The CELLxGENE 45M data used for alignment is also available on Huggingface: https://huggingface.co/arcinstitute/Stack-CellxGene45M. The Parse 10M PBMC data is available at the official website https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-i n-a-single-experiment/#download. All evaluation data used for this project are publicly available; see the Methods section for download details of other datasets. 

Code and model availability. Code for the Stack model is available at https://github.com/ArcInstitute/ stack. Model parameters are available on Huggingface: https://huggingface.co/arcinstitute/Stack-Large, https://huggingface.co/arcinstitute/Stack-Large-Aligned. 

Acknowledgments. We thank Beatrice Bevilacqua, Arshia Nayebnazar, Basak Eraslan, Rishi Verma, Noam Teyssier, Silvana Konermann, Patrick Hsu, and Alexander Dobin for helpful discussions. We thank Theo Roth, Nianzhen Li, Po-Yuan Tung and Yi-Chen Chih for help with experimental validation. This study was supported by Arc Institute. M.D. acknowledges the support from NIH [U54AG076043, U54AG079759, R01DA063148, UM1DA051410]. We also acknowledge the efforts of our colleagues to generate and release large-scale datasets necessary to train and evaluate our models. 

Author contributions. M.D. conceived the Stack project with input from Y.K. and Y.H.R. Y.H.R. and D.P.B. supervised the project, with Y.H.R. coordinating effort across the team. M.D. developed the data loaders, architecture, pre-training, and post-training alignment of the Stack model, and implemented a unified Stack codebase. C.C., R.S., A.A., and M.D. curated pretraining data for the model. M.D. prepared post-training alignment data and curated evaluation data. M.D. and Y.H.R. designed evaluation metrics and layout of results. M.D. and A.A. designed and ran baseline embedding and perturbation models. M.D. performed analysis in the manuscript and visualized results. D.G. analyzed Stack gene modules, improved the codebase, and contributed to the inference scheme for the post-trained Stack model. C.R.T. and Y.R. created visuals for main text figures. M.D. wrote the first draft of the manuscript. All authors wrote the final draft of the manuscript. 

Competing interests. D.G. acknowledges outside interest as part of the founding team of the Autoscience Institute. D.P.B. acknowledges outside interest as a Google Advisor. Y.H.R. is a scientific advisory board member at QureXR. All other authors declare no competing interests. 

# References



A. K. Adduri, D. Gautam, B. Bevilacqua, A. Imran, R. Shah, M. Naghipourfar, N. Teyssier, R. Ilango, S. Nagaraj, M. Dong, et al. Predicting cellular responses to perturbation across diverse contexts with STATE. bioRxiv, pages 2025–06, 2025. 





C. Ahlmann-Eltze, W. Huber, and S. Anders. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. Nature Methods, pages 1–5, 2025. 





R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill, et al. On the opportunities and risks of foundation models. arXiv e-prints, pages arXiv–2108, 2021. 





C. Bunne, Y. Roohani, Y. Rosen, A. Gupta, X. Zhang, M. Roed, T. Alexandrov, M. AlQuraishi, P. Brennan, D. B. Burkhardt, et al. How to build the virtual cell with artificial intelligence: Priorities and opportunities. Cell, 187(25):7045–7063, 2024. 





T. T. S. Consortium*, R. C. Jones, J. Karkanias, M. A. Krasnow, A. O. Pisco, S. R. Quake, J. Salzman, N. Yosef, B. Bulthaup, P. Brown, et al. The tabula sapiens: A multiple-organ, single-cell transcriptomic atlas of humans. Science, 376(6594):eabl4896, 2022. 





H. Cui, C. Wang, H. Maan, K. Pang, F. Luo, N. Duan, and B. Wang. scgpt: toward building a foundation model for single-cell multi-omics using generative ai. Nature Methods, 21(8):1470–1480, 2024. 





I. H. De Boer, C. E. Alpers, E. U. Azeloglu, U. G. Balis, J. M. Barasch, L. Barisoni, K. N. Blank, A. S. Bomback, K. Brown, P. C. Dagher, et al. Rationale and design of the kidney precision medicine project. Kidney international, 99(3):498–510, 2021. 





J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171– 4186, 2019. 





J. Ding, J. Lin, S. Jiang, Y. Wang, Z. Miao, Z. Fang, J. Tang, M. Li, and X. Qiu. Toward a privacy-preserving predictive foundation model of single-cell transcriptomics with federated learning and tabular modeling. bioRxiv, pages 2025–01, 2025. 





M. Dong, B. Wang, J. Wei, A. H. de O. Fonseca, C. J. Perry, A. Frey, F. Ouerghi, E. F. Foxman, J. J. Ishizuka, R. M. Dhodapkar, et al. Causal identification of single-cell experimental perturbation effects with cinema-ot. Nature methods, 20(11):1769–1779, 2023. 





M. Dong, K. Agrawal, R. Fan, E. Sefik, R. A. Flavell, and Y. Kluger. Scaling deep identifiable models enables zero-shot characterization of single-cell biological states. bioRxiv, pages 2023–11, 2024. 





M. Dong, L. Wang, and Y. Kluger. Understanding and enhancing mask-based pretraining towards universal representations. The Thirty-Ninth Annual Conference on Neural Information Processing Systems, 2025. 





R. D. Edgar, D. Nakib, D. Camat, S. Chung, P. Lumanto, J. Atif, C. T. Perciani, X.-Z. Ma, C. Thoeni, N. Selvakumaran, et al. A single-cell atlas of human pediatric liver reveals age-related hepatic gene signatures. bioRxiv, pages 2025–04, 2025. 





M. I. Gabitto, K. J. Travaglini, V. M. Rachleff, E. S. Kaplan, B. Long, J. Ariza, Y. Ding, J. T. Mahoney, N. Dee, J. Goldy, et al. Integrated multimodal cell atlas of alzheimer’s disease. Nature Neuroscience, 27(12):2366– 2383, 2024. 





A. Gayoso, R. Lopez, G. Xing, P. Boyeau, V. Valiollah Pour Amiri, J. Hong, K. Wu, M. Jayasuriya, E. Mehlman, M. Langevin, et al. A python library for probabilistic analysis of single-cell omics data. Nature biotechnology, 40(2):163–166, 2022. 





J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020. 





M. Hao, J. Gong, X. Zeng, C. Liu, Y. Guo, X. Cheng, T. Wang, J. Ma, X. Zhang, and L. Song. Large-scale foundation model on single-cell transcriptomics. Nature methods, 21(8):1481–1491, 2024. 





K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick. Masked autoencoders are scalable vision learners. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022. 





N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister, and F. Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326, 2025. 





A. C. Huang, T.-H. S. Hsieh, J. Zhu, J. Michuda, A. Teng, S. Kim, E. M. Rumsey, S. K. Lam, I. Anigbogu, P. Wright, et al. X-atlas/orion: Genome-wide perturb-seq datasets via a scalable fix-cryopreserve platform for training dose-dependent biological foundation models. bioRxiv, pages 2025–06, 2025. 





G. D. Kalliolias and L. B. Ivashkiv. TNF biology, pathogenic mechanisms and emerging therapeutic strategies. Nature reviews rheumatology, 12(1):49–62, 2016. 





K. Z. Kedzierska, L. Crawford, A. P. Amini, and A. X. Lu. Zero-shot evaluation reveals limitations of single-cell foundation models. Genome Biology, 26(1):101, 2025. 





E. Kernfeld, Y. Yang, J. S. Weinstock, A. Battle, and P. Cahan. A systematic comparison of computational methods for expression forecasting. BioRxiv, pages 2023–07, 2023. 





I. Khemakhem, D. Kingma, R. Monti, and A. Hyvarinen. Variational autoencoders and nonlinear ica: A unifying framework. In International conference on artificial intelligence and statistics, pages 2207–2217. PMLR, 2020. 





D. Klein, J. S. Fleck, D. Bobrovskiy, L. Zimmermann, S. Becker, A. Palma, L. Dony, A. Tejada-Lapuerta, G. Huguet, H.-C. Lin, et al. Cellflow enables generative single-cell phenotype modeling with flow matching. bioRxiv, pages 2025–04, 2025. 





K. D. Koh, L. R. Bonser, W. L. Eckalbar, O. Yizhar-Barnea, J. Shen, X. Zeng, K. L. Hargett, D. I. Sun, L. T. Zlock, W. E. Finkbeiner, et al. Genomic characterization and therapeutic utilization of il-13-responsive sequences in asthma. Cell genomics, 3(1), 2023. 





J. Lause, P. Berens, and D. Kobak. Analytic pearson residuals for normalization of single-cell rna-seq umi data. Genome biology, 22(1):258, 2021. 





C. Lee, M. An, J.-G. Joung, W.-Y. Park, D. K. Chang, Y.-H. Kim, and S. N. Hong. Tnf?? induces lgr5+ stem cell dysfunction in patients with crohn’s disease. Cellular and Molecular Gastroenterology and Hepatology, 13 (3):789–808, 2022. 





C. Li, H. Gao, Y. She, H. Bian, Q. Chen, K. Liu, L. Wei, and X. Zhang. Benchmarking ai models for in silico gene perturbation of cells. bioRxiv, pages 2024–12, 2024. 





X. Li, K. Singhal, Q. Deng, D. Chihara, D. Russler-Germain, R. A. Harkins, J. Henderson, K. Arita, A. Kizhakeyil, R. Sun, et al. Large b cell lymphoma microenvironment archetype profiles. Cancer Cell, 2025. 





T. Liu, K. Li, Y. Wang, H. Li, and H. Zhao. Evaluating the utilities of foundation models in single-cell data analysis. bioRxiv, pages 2023–09, 2023. 





S. Longpre, L. Hou, T. Vu, A. Webson, H. W. Chung, Y. Tay, D. Zhou, Q. V. Le, B. Zoph, J. Wei, and A. Roberts. The flan collection: Designing data and methods for effective instruction tuning, 2023. 





R. Lopez, J. Regier, M. B. Cole, M. I. Jordan, and N. Yosef. Deep generative modeling for single-cell transcriptomics. Nature methods, 15(12):1053–1058, 2018. 





M. D. Luecken, M. Büttner, K. Chaichoompu, A. Danese, M. Interlandi, M. F. Müller, D. C. Strobl, L. Zappia, M. Dugas, M. Colomé-Tatché, et al. Benchmarking atlas-level data integration in single-cell genomics. Nature methods, 19(1):41–50, 2022. 





M. D. Luecken, S. Gigante, D. B. Burkhardt, R. Cannoodt, D. C. Strobl, N. S. Markov, L. Zappia, G. Palla, W. Lewis, D. Dimitrov, et al. Defining and benchmarking open problems in single-cell analysis. Nature Biotechnology, pages 1–6, 2025. 





M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 





L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe. Training language models to follow instructions with human feedback, 2022. 





Parse Biosciences. 10 million human pbmcs in a single experiment, 2023. URL https://www.parsebioscienc es.com/datasets/10-million-human-pbmcs-in-a-single-experiment/. 





J. D. Pearce, S. E. Simmonds, G. Mahmoudabadi, L. Krishnan, G. Palla, A.-M. Istrate, A. Tarashansky, B. Nelson, O. Valenzuela, D. Li, et al. A cross-species generative cell atlas across 1.5 billion years of evolution: The transcriptformer single-cell model. bioRxiv, pages 2025–04, 2025. 





C. C. S. Program, S. Abdulla, B. Aevermann, P. Assis, S. Badajoz, S. M. Bell, E. Bezzi, B. Cakir, J. Chaffer, S. Chambers, et al. CZ CELLxGENE Discover: a single-cell data platform for scalable exploration, analysis and modeling of aggregated data. Nucleic Acids Research, 53(D1):D886–D900, 2025. 





J. Qu, D. HolzmÃžller, G. Varoquaux, and M. L. Morvan. Tabicl: A tabular foundation model for in-context learning on large data. arXiv preprint arXiv:2502.05564, 2025. 





P. Rautenstrauch and U. Ohler. Shortcomings of silhouette in single-cell integration benchmarking. Nature Biotechnology, pages 1–5, 2025. 





Y. H. Roohani, T. J. Hua, P.-Y. Tung, L. R. Bounds, F. B. Yu, A. Dobin, N. Teyssier, A. Adduri, A. Woodrow, B. S. Plosky, et al. Virtual cell challenge: Toward a turing test for the virtual cell. Cell, 188(13):3370–3374, 2025. 





Y. Rosen, Y. Roohani, A. Agarwal, L. Samotorčan, T. S. Consortium, S. R. Quake, and J. Leskovec. Universal cell embeddings: A foundation model for cell biology. bioRxiv, pages 2023–11, 2023. 





S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. Chiu, A. Rush, and V. Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136– 130184, 2024. 





Y. Saito, M. Shimizu, K. Iwatsuki, H. Hanyu, M. Tadaishi, Y. Sugita-Konishi, and K. Kobayashi-Hattori. Effect of short-time treatment with tnf-?? on stem cell activity and barrier function in enteroids. Cytotechnology, 73(4):669–682, 2021. 





S. Salcher, G. Sturm, L. Horvath, G. Untergasser, C. Kuempers, G. Fotakis, E. Panizzolo, A. Martowicz, M. Trebo, G. Pall, et al. High-resolution single-cell atlas reveals diversity and plasticity of tissue-resident neutrophils in non-small cell lung cancer. Cancer cell, 40(12):1503–1520, 2022. 





L. Sikkema, C. Ramírez-Suástegui, D. C. Strobl, T. E. Gillett, L. Zappia, E. Madissoon, N. S. Markov, L.-E. Zaragosi, Y. Ji, M. Ansari, et al. An integrated cell atlas of the lung in health and disease. Nature medicine, 29(6):1563–1577, 2023. 





W. R. Swindell, M. A. Beamer, M. K. Sarkar, S. Loftus, J. Fullmer, X. Xing, N. L. Ward, L. C. Tsoi, M. J. Kahlenberg, Y. Liang, et al. Rna-seq analysis of il-1b and il-36 responses in epidermal keratinocytes identifies a shared myd88-dependent gene signature. Frontiers in immunology, 9:80, 2018. 





C. V. Theodoris, L. Xiao, A. Chopra, M. D. Chaffin, Z. R. Al Sayed, M. C. Hill, H. Mantineo, E. M. Brydon, Z. Zeng, X. S. Liu, et al. Transfer learning enables predictions in network biology. Nature, 618(7965): 616–624, 2023. 





J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le. Finetuned language models are zero-shot learners, 2022. 





S. B. Wells, D. B. Rainbow, M. Mark, P. A. Szabo, C. Ergen, D. P. Caron, A. R. Maceiras, E. Rahmani, E. Benuck, V. Valiollah Pour Amiri, et al. Multimodal profiling reveals tissue-directed signatures of human immune cells altered with age. Nature Immunology, pages 1–14, 2025. 





Y. Wu, E. Wershof, S. M. Schmon, M. Nassar, B. Osiński, R. Eksi, K. Zhang, and T. Graepel. Perturbench: Benchmarking machine learning models for cellular perturbation analysis. arXiv preprint arXiv:2408.10609, 2024. 





N. Youngblut, C. Carpenter, A. Nayebnazar, A. Adduri, R. Shah, C. Ricci-Tam, J. Prashar, R. Ilango, N. Teyssier, S. Konermann, et al. scBaseCount: an AI agent-curated, uniformly processed, and autonomously updated single cell data repository. 2025. 





J. Zhang, A. A. Ubas, R. de Borja, V. Svensson, N. Thomas, N. Thakar, I. Lai, A. Winters, U. Khan, M. G. Jones, J. D. Thompson, V. Tran, J. Pangallo, E. Papalexi, A. Sapre, H. Nguyen, O. Sanderson, M. Nigos, O. Kaplan, S. Schroeder, B. Hariadi, S. Marrujo, C. C. A. Salvino, G. Gallareta Olivares, R. Koehler, G. Geiss, A. Rosenberg, C. Roco, D. Merico, N. Alidoust, H. Goodarzi, and J. Yu. Tahoe-100m: A giga-scale single-cell perturbation atlas for context-dependent gene function and cellular modeling. bioRxiv, pages 2025–02, 2025. 





J. Zhou, C. Wei, H. Wang, W. Shen, C. Xie, A. Yuille, and T. Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021. 



# Supplementary Figures and Tables


A


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/59708109d141bf3d1d817947ab4e6fb1e845673b9800bf30540e32ff9388cd56.jpg)



Mean Absolute Error (MAE)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/024897fe7b98e0401ffb7c10e9c07edea8224f94f564a0d0cbc9c5211c094e75.jpg)



Pearson r


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/7fc2b1d751ec203f2f361b2704bc9257ffa24e2cffca5751a528f79f7ba630f1.jpg)



B


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/f6d33c0671f3d318ccb11963f3428928aea6c9f4db3397e526357c9cd5e613cf.jpg)



Mean Absolute Error (MAE)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/1c1020cd13dfd6c7b10585e97e4c522195b657ca3da2bcb7e72d8933eed71937.jpg)



Pearson r


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/d08c0c5f786b43825fce9f9d2c732ea239ff1633a666569b1a0ce5b04fce1455.jpg)



C


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/779d1a9efb98213676e21de0d348125ce6ab3076a2e1994dcafb16af726f47ac.jpg)



Mean Absolute Error (MAE)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/e43b9b445975b43550e6c6c5d5886bf942718c1af435e35f131d3953b85053b2.jpg)



Pearson r


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/bdeeeb70ea20b4a1eb2fc287276ff585000759b426e02266a28aa67228700f08.jpg)



D


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/58095670d19ef462a6123a1fd441f9e4edb5c1ae113cea3c3a3d621be71adc2d.jpg)



Mean Absolute Error (MAE)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/17e6edf64891b4d77a099c885f1a45946f71817144ad3e946f3e4bb73225403b.jpg)



Pearson r


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/11034fd8f622d442413f2c2d311a3ab4e44e9ec6c32f89d89d652f08c901d52d.jpg)



Figure S1 | Scaling and ablation analysis of the Stack model. A. Validation performance across model sizes for Stack models trained on the full human scBaseCount (Youngblut et al., 2025). B. Validation performance across model sizes for Stack models trained on the scBaseCount subset. C. Validation performance across cell set sizes for the Stack (Large) model trained on the full scBaseCount dataset. D. Validation performance across ablation settings for Stack (Base) models trained on scBaseCount subset (w.o. latent reg: removing only latent regularization; w.o. cell attn: removing both latent regularization and inter-cellular attention). All models in A, B and D use a cell set size of 256. For validation loss and mean absolute error (MAE), smaller is better; for Pearson r, higher is better.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/29f0e926c9281894c3436520cbb29938ac5194a7987f6a927a607528a148d8ed.jpg)



Figure S2 | Adjusted ??-value heatmap of Gene Ontology (GO) biological process enrichment analysis for the top 10 most important genes within each Stack (Large) token after pre-training on full human scBaseCount. Gene importance scores were computed by first reshaping the tokenization weight matrix W to ℝ??hidden×??token×??genes , then calculating the mean absolute weight across the token dimension for each hidden module. The top 10 genes per module were selected based on these importance scores. The top 2 enriched pathways per module are shown. Rows (modules) and columns (pathways) were ordered by hierarchical clustering using average linkage with Euclidean distance.



A



Disease and physiological condition probing performance on observational scRNA-seq data (linear probing, original metrics)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/8c500bcb9802e3fe34a3bed3b918b04346d55346a0d2c0a1650fc0f9785443dc.jpg)



B



Disease and physiological condition probing performance on observational scRNA-seq data (MLP probing, original metrics)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/df1e23341083b8129b36db70d0eb956011bd8eec1202c582bdbe63c2c3e7ebac.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/ddb146e009c65284878e537b8b0a39321ca1bf5d47036843d4dd967e8a5715ee.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/232d80e0972165988853978ae79c9067d194707a243d0c07f4832e21d8eb3db9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/31c8ec6ed9039ae33cc640989d1f7cea80e2b923384df41d31de808f6f91daac.jpg)



C



Cell type classification performance on observational scRNA-seq data


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/f9396673ad2b7a25348e4d08ea1c0155e046bc1d85d4b2137c48cfd8c49460fe.jpg)



Figure S3 | Additional probing evaluation results. A. Disease and physiological condition probing performance on observational scRNA-seq data (De Boer et al., 2021; Li et al., 2025; Salcher et al., 2022; Gabitto et al., 2024). One linear classifier is trained per cell type. The number of experiments per dataset: n=12, 20, 10, 20. See Fig. S4 for full results. All Stack results presented here are based on one model with the (Large) setting pretrained on full human scBaseCount. All panels show (average) values of original metrics (balanced accuracy/Pearson r) without normalization. B. Disease and physiological condition MLP probing performance on observational scRNA-seq data. One MLP classifier is trained simultaneously on the top 5 most abundant cell types. See Methods for dataset statistics. C. Cell type classification performance on observational scRNA-seq data. In Tabula Sapiens evaluation, each point represents a tissue (n=26).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/f762406bd8c342be6d5a777705e55631bc83318261dc7a3faa6ac20a8bfeed76.jpg)



Figure S4 | Per-cell-type linear probing results with additional Stack settings. BC: the model is trained on full human scBaseCount. CxG: the model is trained on CELLxGENE. All Context: Stack uses the default dataset grouping by sample, utilizing all cell types from each sample as context. CT Context: Stack utilizes cells grouped by cell type per sample as context, generates a set of embeddings for each group, then concatenates them to form the total embedding. Full shuffle: The cell order in the evaluation data is randomly shuffled, effectively removing context information. The number of experiments per dataset: n=12, 10, 20, 20, 6, 17, 20, 20, 20, 2.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/a5ce5a054c1d15a31ac58a03b85678071e1c7d648a31020b9615156fb95636f4.jpg)



Figure S5 | Linear probing results for each dataset, using the overall best-performing cell type per task. The full results for Xaira are included in Fig. 2.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/627eb1335b949b4c97723fb963f0df72900d04e000531943e132d416bd09c6a5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/57696a83a4e1ea449d54ec0ac305a8fe339c3045765da2698f4948eef29a7ea5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/b03b689317da46ec201a90ecfd255c991e5128065e3cbfbbb98439c4b93f660b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/36b98a61a0f0b59d53e00ec54dea18f558bee2f007ef216c16de0fa70bc6ecdd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/561c07f7b32e67310c3eb67afb99fbf1b41378cf3e94ca4e519da680c579ffe2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/99af28c796494335df6c5e69a02ab718aa26d6820341a3fbb5d8c43491d0a8bc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/4f4719337a6fd90a676f735153f5675888631b7aaf77862f72f234fd7e2215a6.jpg)



Figure S6 | Additional evaluation results of Stack on batch integration. A. scIB evaluation of different methods on integrating donor profiles and preserving cell types (Luecken et al., 2022). Presented Stack results are based on one model with the (Large) setting pretrained on full human scBaseCount.. B. scIB batch integration total scores of Stack with different sizes, training data, and evaluation datasets. C. UMAP visualization of Stack embedding of the Kidney atlas, colored by data collection and fine-grained cell type. D. Comparison of integration performance across different numbers of principal components and scVI latent dimensions. Based on these results, the results of 50 principal components and 30 scVI latent dimensions were selected for main batch integration benchmarks. E. Comparison of dataset label integration performance. As BCL involves only a single dataset label, it is not applicable to dataset label integration evaluation. Alternative foundation models (scGPT, UCE, State (SE), and TranscriptFormer) were excluded from the Tabula Sapiens evaluation due to their limited performance improvements in the per-tissue benchmark. In B, D, and E, we apply Stack to the full dataset rather than to one sample at a time. This results in a very minor decrease in Stack ’s performance. The dataset integration performance in E closely matches the donor integration performance shown in A.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/4ac9184852497943d8426bd0a14a19dcad13fa993fa46d0a84e3fd84a1edc3a5.jpg)



Figure S7 | Additional metrics on cell prompting tasks. A. Evaluation of perturbation effect prediction across cell types on the Dong et al. (2023) cytokine perturbation dataset (6 cytokines). B. Evaluation of perturbation effect prediction across cell types on the OpenProblems drug perturbation dataset (Luecken et al., 2025) (12 drugs). C. Evaluation of T cell response prediction across samples (7 cytokine stimulation conditions). D. Evaluation of donor-specific gene expression generation across five atlases. E. Evaluation of condition-specific expression generation across four PBMC atlases. See Fig. 3 caption for additional details. For all scores, higher values indicate better performance.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/7ce081c2079e5528613bc1dbdc4060faac6b29141a5ab4d4edfacf1805a1678f.jpg)



Figure S8 | Comparison of perturbation response prediction in novel cell types between Stack post-trained from pretrained weights and Stack trained from scratch. A. Results on the Dong et al. (2023) dataset. B. Results on the OpenProblems drug perturbation dataset (Luecken et al., 2025). All percentages shown represent the average performance improvement of Stack over Stack post-trained from scratch.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/0a43916f76381f1272c8b13cc1c9d85b53cacf033bd08ab4514471582c3014bc.jpg)



B Setting 1, OpenProblems drug perturbations


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/0b84e5fb1eca070275311755cd948bff814b6a0bee6707eb268aacb6faa402cd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/29af96cc1aa532ebfd73ebcdc18e271e1ee118108111a883ce30529ba25f0f89.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/099195fc5c86203e8fe6882da60d3a20e8dd17aabc9b4638139ed85ff8d58851.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/564c460089a286795c8f086d6f98118ae04ccf2b3cb39786711a35053e1821e2.jpg)



C Setting 2, Parse cytokine perturbation prompts + Dong et al. control queries (individual cytokines)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/fe81d4bb7d4c9d576e184933f1840a39fce5db22afeac472f022304b352823ab.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/b7c61f3cb227fac434eed77bb2ceedd4fd61945489e0b672c3a33f6a6177fff9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/7d1eddb2b6b1018c008ff566375ee27908ab9fc3af360227016985f3251e35e0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/aa76fe0222defe25252deea47caa97a3d32d6134487d209c2a0d476332f614a2.jpg)



D Setting 3, 4 observational scRNA-seq atlases


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/b50fa3138dc499678eb28587f2151868b1e8d2d1dc060ddd077f92624a28830e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/4b7b56883739363d07bb68339a23fca95d33f5fce5e6559bfbfb1a0d2059393c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/4a58b2d7235aea7dfb5fe2655fe7db955386a219f560fb582a5b3c510810f0a8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/783dd94a8aacf47f6fd4d293230b0200c8744612ca4b8cafcfe7193a5f1c15ef.jpg)



E Setting 4, 4 immune atlas prompts + Tabula Sapiens query


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/2e7f3f660bc913cec883d0c415ec270555cebc5fae0acf7f2698593105a8a0dd.jpg)



Figure S9 | Evaluations of Stack generative procedure on cell prompting tasks. A. Comparison of Stack (T=5) and Stack (T=1) for perturbation effect prediction across cell types on the Dong et al. (2023) dataset. B. Comparison of Stack (T=5) and Stack (T=1) for perturbation effect prediction across cell types on the OpenProblems drug perturbation dataset (Luecken et al., 2025). C. Evaluation of T cell perturbation response prediction across Parse PBMC and Dong et al. (2023) with individual cytokine stimulation conditions. D. Evaluation of donor-specific gene expression generation across five atlases. E. Evaluation of condition-specific expression generation across four PBMC atlases as prompts and Tabula Sapiens as queries. All percentages shown represent the average performance improvement of Stack generative settings over predictive settings. See the Fig. 3 caption for the number of experiments and the definition of scores. For all scores, higher values indicate better performance.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/9ae7f1fdfdf0bd9c7303a00b581ad05bcdb17ef849635b173f1cbf93f4c1e447.jpg)



Figure S10 | Additional analysis on Perturb Sapiens. A. UMAP visualization of Stack embedding of Dactolisib Perturb Sapiens, colored by tissue and cell class label. B. Violin plots of classifier predicted logit value in ADSF Perturb Sapiens, grouped by tissue and cell class. C. Violin plots of classifier predicted logit value in Dactolisib Perturb Sapiens, grouped by tissue and cell class. Lower logit indicates higher generation confidence.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/a4554e18faa4514bdfdaa7ac291d32c43ad5e620760c26ac38163056624ce013.jpg)



Figure S11 | Log2-fold-change heatmap of Dactolisib Perturb Sapiens versus control. Only significantly changed genes are shown in color, the rest are colored gray.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/90581de0a23aa4390f4f8fa88244d690bc366598723cb0c05a4fb6b89fe1d1b9.jpg)



Figure S12 | Log2-fold-change heatmap of Proscillaridin-A Perturb Sapiens versus control. Only significantly changed genes are shown in color, the rest are colored gray.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/ecbebed938e0a3a30d3eaf98c83fc624bd667b7f6593ac68b8b8768b5faebde8.jpg)



Figure S13 | Log2-fold-change heatmap of Ketoconazole Perturb Sapiens versus control. Only significantly changed genes are shown in color, the rest are colored gray.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/7b297d381232d0c710fe231bbfe9167ee4d3d3c0ed1dfd479fc93fe7879b38b2.jpg)



Figure S14 | Comparison of Perturb Sapiens immune cells with biological replicates. A. Evaluation on Parse cytokine perturbations using Donor 2 as the biological replicate. B. Scatter plot of differential expression (DE) overlap accuracy between Perturb Sapiens and the biological replicate for each cytokine (n=90). C. Evaluation on OpenProblems drug perturbations using both remaining donors as biological replicates due to limited cell numbers per individual donor. D. Scatter plot of DE overlap accuracy between Perturb Sapiens and biological replicates for each drug (n=111).


Full generated data w.o. classifier-guided filtering 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/973c0773370067b94046fb3af9d0b8f3e9890d13ebcad6bff6936234df5e5f62.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/6e38d2c5d0d258d7ee7198698ec8dde0c77442217b428c55693cee0a88b93efb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/1a74c89b9d50d54839b3488c2f5f2cedf277a7fa09c573327cc40d1a329b5da2.jpg)



Figure S15 | Results of Perturb Sapiens without classifier-guided filtering. A. Evaluation of Perturb Sapiens epithelial interferon-beta (IFN-??) effects using single-cell IFN-?? stimulation data from primary airway epithelial cells (Koh et al., 2023). B. Evaluation of Perturb Sapiens epithelial interleukin-13 (IL-13) effects using single-cell IL-13 stimulation data from primary airway epithelial cells (Koh et al., 2023). C. Evaluation of Perturb Sapiens epithelial interleukin-1 beta (IL-1??) effects using bulk IL-1?? stimulation data from primary keratinocytes (Swindell et al., 2018).



A Perturb Sapiens per-cell-class IL-17 effects v.s. primary airway epithelial cell IL-17 stimulation scRNA-seq data


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/e53c0bfda0b3f645e3e05e377cdebec32bbba3ec7e08bfe9f8e63e8d00917701.jpg)



B Perturb Sapiens per-cell-class TNF-α effects v.s. intestinal organoid TNF-α stimulation bulk RNA-seq data


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/e402c6f7605d5d191ac84ff6ca9849619163ea06c0cbeee45ce706ea8d88df0a.jpg)



Figure S16 | Results of Perturb Sapiens on additional cytokines. A. Evaluation of Perturb Sapiens epithelial IL-17 effects using single-cell IL-17 stimulation data from primary airway epithelial cells (Koh et al., 2023). B. Evaluation of Perturb Sapiens epithelial TNF-?? effects using bulk TNF-?? stimulation data from primary intestinal epithelial cells (Lee et al., 2022).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/ac9ec5e79704b319b51dc1b3b9205a9df2b553137e1a738ffeff999806a3a103.jpg)



Figure S17 | Global characterization of Perturb Sapiens. A. Average perturbation similarity across cell classes in Perturb Sapiens. B. Average perturbation similarity for T cells across representative tissues in Perturb Sapiens. Values represent mean Spearman correlations averaged across perturbations using Fisher z-transformation. The top three or two edges by correlation are shown for each node.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/8697fe55c214f159bfa09333a805057ecfa616e6f98f434a498dc790bafc3bf2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/5477af407db255c9c20fa57d94952b961d0f43aef23b36cee25dedf4e24cfcbb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/8c96323ff17c3ab3d9a6f7fda743fa175ed2c92bb05dbe30eb1d2893297ff9df.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/8d3c6ab1-237a-49ce-a2ab-fde6cfdd226f/574f284ecf72b31037b7fb7b06a712208ed2802984c37bd410c1883b27ee51cb.jpg)



Figure S18 | Heatmap of independent components derived from Perturb Sapiens drug and cytokine log-fold changes (LFCs) in representative cell types and tissues. A. Epithelial Kidney. B. Epithelial lung. C. T cell lung. D. Factor loading for epithelial kidney. E. Factor loading for epithelial lung. F. Factor loading for T cell lung. All analyses used an independent component number of 15.



Table S1 | Additional evaluation of Perturb Sapiens intestinal epithelial cytokine responses against in vitro TNF-?? bulk data (Saito et al., 2021). Significance: $^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ , n.s. not significant.


<table><tr><td>Pearson Delta</td><td>DE Spearman LFC</td><td>DE Direction Match</td><td>PR AUC</td><td>DE Overlap Accuracy</td></tr><tr><td>-0.122***</td><td>-0.077**</td><td>0.434</td><td>0.530</td><td>0.267</td></tr></table>


Table S2 | Representative genes showing consistent (top) and opposite (bottom) trends between Perturb Sapiens, Parse T cells and in vitro TNF-?? data (Lee et al., 2022).


<table><tr><td>Gene</td><td>Perturb Sapiens LFC</td><td>Adj p-value</td><td>Parse T cell LFC</td><td>Adj p-value</td><td>In vitro data LFC</td><td>Adj p-value</td></tr><tr><td colspan="7">Consistent trends between Perturb Sapiens/Parse and Lee et al. (2022)</td></tr><tr><td>MT1M</td><td>-1.257</td><td>&lt; 10-30</td><td>-2.281</td><td>1.99 × 10-3</td><td>-4.115</td><td>6.69 × 10-5</td></tr><tr><td>MT2A</td><td>-2.215</td><td>&lt; 10-30</td><td>-1.294</td><td>&lt; 10-30</td><td>-3.087</td><td>1.74 × 10-5</td></tr><tr><td>MT1E</td><td>-0.900</td><td>&lt; 10-30</td><td>-1.114</td><td>3.66 × 10-3</td><td>-3.360</td><td>3.90 × 10-4</td></tr><tr><td>SOCS3</td><td>-0.463</td><td>&lt; 10-30</td><td>-1.218</td><td>&lt; 10-30</td><td>-2.388</td><td>1.88 × 10-5</td></tr><tr><td>PLAU</td><td>0.314</td><td>2.69 × 10-10</td><td>0.611</td><td>1.56 × 10-8</td><td>1.815</td><td>2.86 × 10-8</td></tr><tr><td>LDLR</td><td>-0.470</td><td>&lt; 10-30</td><td>-0.406</td><td>3.52 × 10-25</td><td>-0.416</td><td>3.07 × 10-2</td></tr><tr><td>B4GALT5</td><td>-0.515</td><td>&lt; 10-30</td><td>-0.501</td><td>&lt; 10-30</td><td>-0.458</td><td>2.95 × 10-4</td></tr><tr><td>ITGAV</td><td>0.715</td><td>&lt; 10-30</td><td>0.425</td><td>1.00 × 10-9</td><td>2.085</td><td>3.63 × 10-22</td></tr><tr><td>MTSS1</td><td>0.850</td><td>&lt; 10-30</td><td>1.082</td><td>&lt; 10-30</td><td>1.825</td><td>2.27 × 10-2</td></tr><tr><td>STOM</td><td>-0.582</td><td>&lt; 10-30</td><td>-1.231</td><td>&lt; 10-30</td><td>-1.784</td><td>7.39 × 10-6</td></tr><tr><td>GSN</td><td>-1.001</td><td>&lt; 10-30</td><td>-0.536</td><td>&lt; 10-30</td><td>-0.962</td><td>1.98 × 10-5</td></tr><tr><td>TGFA</td><td>0.868</td><td>&lt; 10-30</td><td>0.720</td><td>6.09 × 10-21</td><td>0.753</td><td>1.01 × 10-4</td></tr><tr><td colspan="7">Opposite trends between Perturb Sapiens/Parse and Lee et al. (2022)</td></tr><tr><td>NFKBIA</td><td>-0.499</td><td>&lt; 10-30</td><td>-0.727</td><td>&lt; 10-30</td><td>1.822</td><td>3.65 × 10-10</td></tr><tr><td>SOD2</td><td>-0.429</td><td>&lt; 10-30</td><td>-1.131</td><td>&lt; 10-30</td><td>0.647</td><td>2.80 × 10-3</td></tr><tr><td>RHOB</td><td>-0.269</td><td>5.11 × 10-9</td><td>-0.769</td><td>3.26 × 10-13</td><td>0.491</td><td>2.01 × 10-2</td></tr><tr><td>SAT1</td><td>-0.952</td><td>&lt; 10-30</td><td>-0.651</td><td>&lt; 10-30</td><td>1.046</td><td>7.50 × 10-5</td></tr><tr><td>GCH1</td><td>-1.288</td><td>&lt; 10-30</td><td>-1.618</td><td>&lt; 10-30</td><td>0.356</td><td>3.19 × 10-4</td></tr><tr><td>RIPK2</td><td>-1.168</td><td>&lt; 10-30</td><td>-0.651</td><td>&lt; 10-30</td><td>1.065</td><td>2.83 × 10-6</td></tr><tr><td>DRAM1</td><td>-0.631</td><td>&lt; 10-30</td><td>-1.316</td><td>&lt; 10-30</td><td>1.972</td><td>5.94 × 10-12</td></tr><tr><td>ISG15</td><td>-3.814</td><td>&lt; 10-30</td><td>-1.729</td><td>&lt; 10-30</td><td>2.104</td><td>6.28 × 10-10</td></tr><tr><td>IFIT2</td><td>-4.196</td><td>&lt; 10-30</td><td>-1.613</td><td>&lt; 10-30</td><td>2.081</td><td>2.79 × 10-20</td></tr><tr><td>IFIT3</td><td>-3.947</td><td>&lt; 10-30</td><td>-1.506</td><td>&lt; 10-30</td><td>2.588</td><td>4.02 × 10-25</td></tr><tr><td>RSAD2</td><td>-3.591</td><td>&lt; 10-30</td><td>-1.922</td><td>&lt; 10-30</td><td>3.273</td><td>3.34 × 10-8</td></tr><tr><td>IFITM3</td><td>-1.010</td><td>&lt; 10-30</td><td>-1.540</td><td>&lt; 10-30</td><td>1.379</td><td>2.60 × 10-6</td></tr><tr><td>IFI27</td><td>-1.439</td><td>&lt; 10-30</td><td>-2.269</td><td>&lt; 10-30</td><td>2.529</td><td>1.24 × 10-4</td></tr><tr><td>CXCL10</td><td>-3.706</td><td>&lt; 10-30</td><td>-2.775</td><td>&lt; 10-30</td><td>6.025</td><td>8.72 × 10-21</td></tr><tr><td>CXCL11</td><td>-3.528</td><td>&lt; 10-30</td><td>-2.938</td><td>&lt; 10-30</td><td>3.069</td><td>9.81 × 10-25</td></tr><tr><td>TNFSF10</td><td>-3.015</td><td>&lt; 10-30</td><td>-1.674</td><td>&lt; 10-30</td><td>2.432</td><td>1.06 × 10-30</td></tr><tr><td>CD74</td><td>-1.242</td><td>&lt; 10-30</td><td>-0.329</td><td>&lt; 10-30</td><td>3.979</td><td>&lt; 10-30</td></tr></table>