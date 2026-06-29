#!/bin/bash -e
#SBATCH --time 00:40:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH -e error.err

ml python3/3.10.11
pip install -e ../..
python3 ../src/mlquiades/main.py --a ../sample_data --b test0 --d True

python3 ../src/mlquiades/main.py --a ../sample_data --b test1 --f True

python3 ../src/mlquiades/main.py --a ../sample_data --b test2 --d True --f True

python3 ../src/mlquiades/main.py --a ../sample_data --b test3 --g 10

python3 ../src/mlquiades/main.py --a ../sample_data --b test4 --i 20

python3 ../src/mlquiades/main.py --a ../sample_data --b test5 --j 30

python3 ../src/mlquiades/main.py --a ../sample_data --b test6 --k 3

python3 ../src/mlquiades/main.py --a ../sample_data --b test7 --l 2

python3 ../src/mlquiades/main.py --a ../sample_data --b test8 --m 5

python3 ../src/mlquiades/main.py --a ../sample_data --b test9 --n .05

python3 ../src/mlquiades/main.py --a ../sample_data --b test10 --o 20

python3 ../src/mlquiades/main.py --a ../sample_data --b test11 --p 1e-3

python3 ../src/mlquiades/main.py --a ../sample_data --b test12 --q 1e-1

python3 ../src/mlquiades/main.py --a ../sample_data --b test13 --r 12

python3 ../src/mlquiades/main.py --a ../sample_data --b test14 --d True --f True --j 30 --k 5 --l 2 --o 3 --r 5
