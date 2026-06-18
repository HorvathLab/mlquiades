# mlquiades
*2026/03/05*

(pronounced *em-el-key-ah-days*)

This package takes in bulk RNA cancer cell line sequencing (processed from raw fastqs to TPM counts using STAR) data and GDSC1 and GDSC2 IC50 drug sensitivity scores for palbociclib to build and evaluate three machine learning models.

These models include: neural net with hyperband, random forest, and ridge classifier.

There are three options for feature, in this case gene, selection. They include: only CDK4 and CDK6 related genes (the target for palbociclib); only CDK4, CDK6 and cancer genes (COSMIC); and a Pearson correlation method that keeps only the genes that have rho>=.3 value with the IC50 score in the training dataset only.

<img src='https://github.com/HorvathLab/mlquiades/blob/e8f528ef838218006ed4c53166f81352a3738482/output/all_tissues/plt_accuracy_all_cdk4_6_genes.png' width=100% height=100%>

## Installation
 
Ubuntu and MacOS are supported. Windows is not currently supported.

```
git init
git clone git@github.com:HorvathLab/mlquiades.git
cd mlquiades
uv venv
# activate the venv: your command may differ for a different shell
source .venv/bin/activate
```

If you use a different package manager, install packages to your environment from `pyproject.toml`

## Usage (default)

If using uv
```
uv python pin 3.10
uv run src/mlquiades/main.py --a sample_data --b test --c tpm_palb_rib_with_metadata.csv --o 50 --r cdk4_6_genes --s cdk4_6_genes.txt --u palbociclib --v gdsc1 --w ic50
```

If not using uv
```
python src/mlquiades/main.py --a sample_data --b test --c tpm_palb_rib_with_metadata.csv --o 50 --r cdk4_6_genes --s cdk4_6_genes.txt --u palbociclib --v gdsc1 --w ic50
```

## Output

```
├── output
│   ├── cdk4_6_cancer_gex
│   │   ├── evaluation_df_acc.csv
│   │   ├── evaluation_df.csv
│   │   ├── evaluation_df_rocauc.csv
│   │   ├── plt_accuracy_all_cdk4_6_cancer.png
│   │   └── plt_rocauc_all_cdk4_6_cancer.png
│   ├── cdk4_6_genes_gex
│   │   ├── evaluation_df_acc.csv
│   │   ├── evaluation_df.csv
│   │   ├── evaluation_df_rocauc.csv
│   │   ├── plt_accuracy_all_cdk4_6_genes.png
│   │   └── plt_rocauc_all_cdk4_6_genes.png
│   ├── data_split.csv
│   ├── data_split.png
│   ├── pearson_gex
│   │   ├── evaluation_df_acc.csv
│   │   ├── evaluation_df.csv
│   │   ├── evaluation_df_rocauc.csv
│   │   ├── plt_accuracy_all_pearson.png
│   │   └── plt_rocauc_all_pearson.png
│   ├── report_gex.html
│   └── report_gex.md
```


## Authors

Horvath Lab + Joe Goldfrank

George Washington University

McCormick Genomic and Proteomic Center (MGPC) & Department of Computer Science

*Questions? siera.martinez@gwu.edu*
