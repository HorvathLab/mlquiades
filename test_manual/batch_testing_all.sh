#!/bin/bash -e
#SBATCH --time 1-00:00:00
#SBATCH --partition defq
##SBATCH -n 40
#SBATCH -e error.err

mkdir testing_1                                                                                                                                     
cd testing_1

ml python3/3.10.11
python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_2
cd testing_2

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 2.5 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_3
cd testing_3

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --f False --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_4
cd testing_4

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --g 10 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_5
cd testing_5

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --i 10 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_6
cd testing_6

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --j 30 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_7
cd testing_7

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --k 2 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_8
cd testing_8

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --m 1 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_9
cd testing_9

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --n .001 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_10
cd testing_10

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --o 2 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_11
cd testing_11

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --p 1e-4 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_12
cd testing_12

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --q 1e-2 --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_13
cd testing_13

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --r cancer_genes --s cdk4_6_genes.txt --t cdk4_6_cancer_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_14
cd testing_14

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 4 --r pearson --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_15
cd testing_15

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 2 --f False --g 10 --i 10 --j 50 --k 2 --l 2 --m 2 --n .001 --o 1 --p 1e-4 --q 1e-2 --r pearson --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz
                                                                                                                                                            cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_16
cd testing_16

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 2 --f False --g 10 --i 10 --j 50 --k 2 --l 2 --m 2 --n .001 --o 1 --p 1e-4 --q 1e-2  --r cdk4_6_cancer_genes --s cdk4_6_genes.txt --t cancer_genes.tsv --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
mkdir testing_17
cd testing_17

python ../../src/mlquiades/main.py --a ../../sample_data --b output --c CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz --d palbociclib.csv --e 2 --f False --g 10 --i 10 --j 50 --k 2 --l 2 --m 2 --n .001 --o 1 --p 1e-4 --q 1e-2  --r cdk4_6_genes --s cdk4_6_genes.txt --u gencode.v19.genes.v7_model.patched_contigs.gtf.gz

cd output
python ../../output_test.py
cd ../../../sample_data/

pigz CCLE_RNAseq_rsem_genes_tpm_20180929.txt
pigz gencode.v19.genes.v7_model.patched_contigs.gtf

cd ../testing/
