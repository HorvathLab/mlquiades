# MLQUIADES
*2025/07/28*

(pronounced *em-el-key-ah-days*)

This package takes in bulk RNA cancer cell line sequencing data and GDSC1 IC50 drug sensitivity scores for palbociclib to build and evaluate 6 machine learning models.

These models include: decision tree, gradient boosted decision tree, neural net (not stable), neural net with hyperband, random forest, and ridge classifier.

There are three options for feature, in this case gene, selection. They include: only CDK4 and CDK6 related genes (the target for palbociclib); only CDK4, CDK6 and cancer genes (COSMIC); and a Pearson correlation method that keeps only the genes that have >=.3 rho value with the IC50 score in the training dataset only.

## Installation
 
Ubuntu and MacOS are upported. Windows is not currently supported.

```
git init
git clone git@github.com:HorvathLab/mlquiades.git
cd mlquiades
```

Setting up Environment (if not using uv)
```
pip install -e .     #installs necessary packages using pyproject.toml; edit pyproject.toml to fit your package versions if necessary
```

## Usage (default)

If using uv
```
uv python pin 3.10
uv run src/mlquiades/main.py --a sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --o 50 --r cdk4_6_genes --s cdk4_6_genes.txt --t cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz
```

If not using uv
```
python src/mlquiades/main.py --a sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --o 50 --r cdk4_6_genes --s cdk4_6_genes.txt --t cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz
```

## Output

```
├── output_dir
│   ├── plt_confusion_dt_cdk_4_6_genes.png
│   ├── plt_confusion_gbdt_cdk_4_6_genes.png
│   ├── plt_confusion_nn_cdk_4_6_genes.png
│   ├── plt_confusion_nn_hb_cdk_4_6_genes.png
│   ├── plt_confusion_rf_cdk_4_6_genes.png
│   ├── plt_confusion_ridge_cdk_4_6_genes.png
│   ├── plt_rocauc_all_cdk_4_6_genes.png
│   ├── plt_rocauc_dt_cdk_4_6_genes.png
│   ├── plt_rocauc_gbdt_cdk_4_6_genes.png
│   ├── plt_rocauc_nn_cdk_4_6_genes.png
│   ├── plt_rocauc_nn_hb_cdk_4_6_genes.png
│   ├── plt_rocauc_rf_cdk_4_6_genes.png
│   ├── plt_rocauc_ridge_cdk_4_6_genes.png
│   ├── report.html
│   └── report.md
```

## Usage (thorough)


The following commands take in the CCLE, palbociclib, and genes.gtf files to build a dataframe for running through the 6 ML models. It selects features (-r), or genes, that are only related to CDK4 and CDK6. It also sets the IC50 cutoff value for palbociclib to 4 (the reported value on cancergenex.org). This randomly oversamples training data by default. The output directory is set to output_dir.

```
python src/mlquiades/main.py --a sample_data --b output_dir --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz
```

The following commands take in the CCLE, palbociclib, and genes.gtf files to build a dataframe for running through the 6 ML models. It selects features (-r), or genes, that are only related to CDK4, CDK6 and cancer. It also sets the IC50 cutoff value for palbociclib to 4 (the reported value on cancergenex.org). This randomly oversamples training data by default. The output directory is set to output_dir.

```
python src/mlquiades/main.py --a sample_data --b output_dir --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --r cdk_4_6_cancer_genes --s cdk4_6_genes.txt --t cancer_genes.tsv --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz
```

The following commands take in the CCLE, palbociclib, and genes.gtf files to build a dataframe for running through the 6 ML models. It selects features (-r), or genes, that have a Pearson correlation rho value of .3 or greater with the y-label values (in the training data only). It also sets the IC50 cutoff value for palbociclib to 4 (the reported value on cancergenex.org). This randomly oversamples training data by default. The output directory is set to output_dir.

```
python src/mlquiades/main.py --a sample_data --b output_dir --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --r pearson --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz
```

To prevent random oversampling of data, a select CDK4 and CDK6 genes as features:

```
python src/mlquiades/main.py --a sample_data --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --f False --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz
```

To modify parameters for hyperband tuning and select CDK4 and CDK6 genes as features, the following command may be used. This sets: the node step size to 20, the minimum number of nodes to 2, the maximum number of nodes to 200, the maximum number of trials to 5, the number of executions per trial to 4, the patience to 5, the minimum delta in early stopping to .1, the number of epochs to 40, the learning rate minimum to .001, and the learning rate maximum to .1.

```
python src/mlquiades/main.py --a sample_data --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --f False --g 20 --i 2 --j 200 --k 5 --l 4 --m 4 --n .1 --o 40 --p .001 --q .1 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz
```

## Authors

Horvath Lab + Joe Goldfrank

George Washington University

McCormick Genomic and Proteomic Center (MGPC) & Department of Computer Science

*Questions? siera.martinez@gwu.edu*
