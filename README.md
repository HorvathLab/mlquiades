# mlquiades
*2026/03/05*

(pronounced *em-el-key-ah-days*)

This package takes in bulk RNA cancer cell line sequencing (processed from raw fastqs to TPM counts using STAR) data and GDSC1 and GDSC2 IC50 drug sensitivity scores for palbociclib to build and evaluate three machine learning models.

These models include: neural net with hyperband, random forest, and ridge classifier.

There are three options for feature, in this case gene, selection. They include: only CDK4 and CDK6 related genes (the target for palbociclib); only CDK4, CDK6 and cancer genes (COSMIC); and a Pearson correlation method that keeps only the genes that have >=.3 rho value with the IC50 score in the training dataset only.

<img src='https://github.com/HorvathLab/mlquiades/blob/e8f528ef838218006ed4c53166f81352a3738482/output/all_tissues/plt_accuracy_all_cdk4_6_genes.png' width=100% height=100%>

## Installation
 
Ubuntu and MacOS are supported. Windows is not currently supported.

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
uv run src/mlquiades/main.py --a sample_data --b output --c isoforms.csv --o 50 --r cdk4_6_genes --s cdk4_6_genes.txt --u palbociclib --v gdsc1
```

If not using uv
```
python src/mlquiades/main.py --a sample_data --b output --c isoforms.csv --o 50 --r cdk4_6_genes --s cdk4_6_genes.txt --u palbociclib --v gdsc1
```

## Output

```
├── output
│   ├── all_tissues
│   │   ├── data_split.csv
│   │   ├── data_split.png
│   │   ├── plt_accuracy_all_cdk4_6_genes.png
│   │   └── plt_rocauc_all_cdk4_6_genes.png
│   ├── report.html
│   └── report.md
```

## Usage (thorough)


The following commands take in cancer cell line TPMs, palbociclib, and genes.gtf files to build a dataframe for running through the 3 ML models. It selects features (-r), or genes, that are only related to CDK4 and CDK6. This randomly oversamples training data by default. The output directory is set to output_dir.

```
python src/mlquiades/main.py --a sample_data --b output --c isoforms.csv --r cdk4_6_genes --s cdk4_6_genes.txt --u palbociclib --v gdsc1
```

The following commands take in the cancer cell line TPMs, palbociclib, and genes.gtf files to build a dataframe for running through the 3 ML models. It selects features (-r), or genes, that are only related to CDK4, CDK6 and cancer. This randomly oversamples training data by default. The output directory is set to output_dir.

```
python src/mlquiades/main.py --a sample_data --b output_dir --c isoforms.csv --r cdk_4_6_cancer_genes --s cdk4_6_genes.txt --t cancer_genes.tsv --u palbociclib --v gdsc1
```

The following commands take in cancer cell line TPMs, palbociclib, and genes.gtf files to build a dataframe for running through the 3 ML models. It selects features (-r), or genes, that have a Pearson correlation rho value of .3 or greater with the y-label values (in the training data only). It also sets the IC50 cutoff value for palbociclib to 4 (the reported value on cancergenex.org). This randomly oversamples training data by default. The output directory is set to output_dir.

```
python src/mlquiades/main.py --a sample_data --b output_dir --c isoforms.csv --d palbociclib_new.csv --e 4 --r pearson --s cdk4_6_genes.txt --u palbociclib --v gdsc1
```

To prevent random oversampling of data, a select CDK4 and CDK6 genes as features:

```
python src/mlquiades/main.py --a sample_data --c isoforms.csv --f False --r cdk4_6_genes --s cdk4_6_genes.txt --u palbociclib --v gdsc1
```

To modify parameters for hyperband tuning and select CDK4 and CDK6 genes as features, the following command may be used. This sets: the node step size to 20, the minimum number of nodes to 2, the maximum number of nodes to 200, the maximum number of trials to 5, the number of executions per trial to 4, the patience to 5, the minimum delta in early stopping to .1, the number of epochs to 40, the learning rate minimum to .001, and the learning rate maximum to .1.

```
python src/mlquiades/main.py --a sample_data --c isoforms.csv --f False --g 20 --i 2 --j 200 --k 5 --l 4 --m 4 --n .1 --o 40 --p .001 --q .1 --r cdk4_6_genes --s cdk4_6_genes.txt --u palbociclib --v gdsc1
```

## Authors

Horvath Lab + Joe Goldfrank

George Washington University

McCormick Genomic and Proteomic Center (MGPC) & Department of Computer Science

*Questions? siera.martinez@gwu.edu*
