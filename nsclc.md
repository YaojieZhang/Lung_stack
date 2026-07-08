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

AnnData object with n_obs × n_vars = 2666754 × 58336
    obs: 'n_genes', 'total_counts', 'doublet_scores', 'predicted_doublets', 'sample_lineage', 'sample', 'batch', 'pct_counts_mt'
    var: 'gene_id', 'gene_type'
                                   n_genes  total_counts  doublet_scores  \
barcodes                                                                   
P187T_AACACACAGACCAGGTCATTAGCATCC     1107        3020.0        0.010878   
P187T_AACACACAGACCGTACTCCGACTGAGA     2707       10997.0        0.036411   
P187T_AACACACAGACCTCGACTAGCACATGC     1632        6590.0        0.222506   
P187T_AACACACAGACCTCGACTCTGTTCGGT      895        2146.0        0.033499   
P187T_AACACACAGACCTGCTACTGCTATCGC      888        1560.0        0.024109   
...                                    ...           ...             ...   
P182T_TGTGGACACTGAGTAGTCGGCAACACT      834        1811.0        0.063116   
P182T_TGTGGACACTGAGTAGTCTCGTCATGC     1572        5214.0        0.016008   
P182T_TGTGGACACTTACACGACCTCTAACAC      800        1331.0        0.021984   
P182T_TGTGGACACTTCGAGGATCCTTAGGTG     1856        4990.0        0.074197   
P182T_TGTGGACACTTGCCGTCAAGGCAGAAC      488         747.0        0.015271   

                                   predicted_doublets sample_lineage sample  \
barcodes                                                                      
P187T_AACACACAGACCAGGTCATTAGCATCC               False          Mesen  P187T   
P187T_AACACACAGACCGTACTCCGACTGAGA               False           Endo  P187T   
P187T_AACACACAGACCTCGACTAGCACATGC               False              T  P187T   
P187T_AACACACAGACCTCGACTCTGTTCGGT               False           Endo  P187T   
P187T_AACACACAGACCTGCTACTGCTATCGC               False              T  P187T   
...                                               ...            ...    ...   
P182T_TGTGGACACTGAGTAGTCGGCAACACT               False           Mast  P182T   
P182T_TGTGGACACTGAGTAGTCTCGTCATGC               False            Epi  P182T   
P182T_TGTGGACACTTACACGACCTCTAACAC               False         Immune  P182T   
P182T_TGTGGACACTTCGAGGATCCTTAGGTG               False          Mesen  P182T   
P182T_TGTGGACACTTGCCGTCAAGGCAGAAC               False         Immune  P182T   

                                   batch  pct_counts_mt  
barcodes                                                 
P187T_AACACACAGACCAGGTCATTAGCATCC  P187T      17.350994  
P187T_AACACACAGACCGTACTCCGACTGAGA  P187T      25.725197  
P187T_AACACACAGACCTCGACTAGCACATGC  P187T      14.415781  
P187T_AACACACAGACCTCGACTCTGTTCGGT  P187T      25.908667  
P187T_AACACACAGACCTGCTACTGCTATCGC  P187T      10.000000  
...                                  ...            ...  
P182T_TGTGGACACTGAGTAGTCGGCAACACT  P182T      11.927113  
P182T_TGTGGACACTGAGTAGTCTCGTCATGC  P182T      10.606061  
P182T_TGTGGACACTTACACGACCTCTAACAC  P182T       5.184072  
P182T_TGTGGACACTTCGAGGATCCTTAGGTG  P182T      10.941884  
P182T_TGTGGACACTTGCCGTCAAGGCAGAAC  P182T       6.827309  

[2666754 rows x 8 columns]
['P187T', 'P9', 'P29', 'P111T', 'P159T', ..., 'P115T', 'P180T', 'P173T', 'P120T', 'P182T']
Length: 179
Categories (179, object): ['P187T', 'P9', 'P29', 'P111T', ..., 'P180T', 'P173T', 'P120T', 'P182T']