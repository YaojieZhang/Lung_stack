NG_NSCLC:
AnnData object with n_obs × n_vars = 124483 × 31272
    obs: 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'sample_id', 'Platform', 'Patient', 'Timepoint', 'PathType', 'response', 'treatment', 'Residual', 'Group', 'Match', 'percent.mt', 'percent.rb', 'housekeeping_score', 'S.Score', 'G2M.Score', 'Phase', 'RNA_snn_res.0.6', 'seurat_clusters', 'cell_type', 'lineage', 'major', 'minor', 'ST_sample', 'ident'
    var: 'gene_symbols'
    obsm: 'X_harmony', 'X_pca', 'X_tsne', 'X_umap'
    layers: 'logcounts'
                  Patient   sample_id   Timepoint cell_type   sample_id  \
index                                                                     
BD_P_P01_N_219824     P01  BD_P_P01_N  Pre-biopsy         T  BD_P_P01_N   
BD_P_P01_N_775406     P01  BD_P_P01_N  Pre-biopsy         T  BD_P_P01_N   
BD_P_P01_N_568296     P01  BD_P_P01_N  Pre-biopsy         T  BD_P_P01_N   
BD_P_P01_N_40126      P01  BD_P_P01_N  Pre-biopsy         T  BD_P_P01_N   
BD_P_P01_N_68304      P01  BD_P_P01_N  Pre-biopsy         T  BD_P_P01_N   

                      treatment response  
index                                     
BD_P_P01_N_219824  Camrelizumab     NMPR  
BD_P_P01_N_775406  Camrelizumab     NMPR  
BD_P_P01_N_568296  Camrelizumab     NMPR  
BD_P_P01_N_40126   Camrelizumab     NMPR  
BD_P_P01_N_68304   Camrelizumab     NMPR  
Patient value counts:
<bound method Series.sort_index of Patient
P10    24998
P06    22740
P05    21308
P01    19773
P08    17719
P07    15213
P09     2732
Name: count, dtype: int64>
sample_id value counts:
<bound method Series.sort_index of sample_id
XGY_S_P05_P    15472
XGY_P_P06_M    14843
XGY_P_P10_N    12630
XGY_S_P10_N    12368
XGY_P_P08_N    12081
XGY_P_P07_N    10312
XGY_S_P06_M     7897
XGY_S_P01_N     7567
BD_S_P01_N      6599
XGY_P_P05_P     5836
XGY_S_P08_N     5638
BD_P_P01_N      5607
XGY_S_P07_N     4901
XGY_P_P09_P     2339
XGY_S_P09_P      393
Name: count, dtype: int64>
Timepoint value counts:
<bound method Series.sort_index of Timepoint
Pre-biopsy      63648
Post-surgery    60835
Name: count, dtype: int64>
cell_type value counts:
<bound method Series.sort_index of cell_type
Epithelium     40320
T              23007
Mono/Macro     20604
Plasma         13363
B              12571
Endothelium     4469
DC              4216
Pericyte        1947
Fibroblast      1880
Mast            1871
Neutrophil       235
Name: count, dtype: int64>
sample_id value counts:
<bound method Series.sort_index of sample_id
XGY_S_P05_P    15472
XGY_P_P06_M    14843
XGY_P_P10_N    12630
XGY_S_P10_N    12368
XGY_P_P08_N    12081
XGY_P_P07_N    10312
XGY_S_P06_M     7897
XGY_S_P01_N     7567
BD_S_P01_N      6599
XGY_P_P05_P     5836
XGY_S_P08_N     5638
BD_P_P01_N      5607
XGY_S_P07_N     4901
XGY_P_P09_P     2339
XGY_S_P09_P      393
Name: count, dtype: int64>
treatment value counts:
<bound method Series.sort_index of treatment
Camrelizumab     77703
Toripalimab      22740
Sintilimab       21308
Pembrolizumab     2732
Name: count, dtype: int64>
response value counts:
<bound method Series.sort_index of response
NMPR    77703
pCR     24040
MPR     22740
Name: count, dtype: int64>
         gene_symbols
index                
A1BG             A1BG
A1BG-AS1     A1BG-AS1
A1CF             A1CF
A2M               A2M
A2M-AS1       A2M-AS1



P01 ['NMPR']

P05 ['pCR']

P06 ['MPR']

P07 ['NMPR']

P08 ['NMPR']

P09 ['pCR']

P10 ['NMPR']


HJJ_NSCLC:

 2,666,754 cells
178 samples
pairs of samples:
ICT: 16 pairs of pre & post
IAT: 25 pairs of pre & post
CT: 1 pair of pre & post
pre-treatment biopsy samples:
ICT: 26 samples (21 with response labels)
IAT: 29 samples (27 with response labels)
CT: 1 samples with label

print(adata_IAT.obs.groupby('sample_id').size())
P4     10680
P14    24916
P16    20814
P20    24394
P25     8904
P34     9607
P35    14619
P37    17092
P38     2815
P39    14443
P40     8781
P41    13601
P42    14976
P43     9241
P44    11908
P45    14996
P46    12988
P47    13180
P48    16279
P49    14471
P50     9704
P52    14731
P53    17555
P54    12266
P55    12327
P56    17649
P57    14656

print(adata_ICT.obs.groupby('sample_id').size())
sample_id
BD_P_P01_N      5607
BD_immune05     3113
BD_immune08     5825
P1             18704
P2             12855
P3             11162
P5             22571
P6             13472
P7             18481
P8             22983
P9              4180
P10             5793
P11            17999
P12            19394
P15             7164
P18            18556
P19            14181
P21            16211
P22            13077
P27            13644
P29             9360
P30             9951
P31             5621
P33            12152
XGY_P_P05_P     5836
XGY_P_P06_M    14843
XGY_P_P07_N    10312
XGY_P_P08_N    12081
XGY_P_P09_P     2339
XGY_P_P10_N    12630