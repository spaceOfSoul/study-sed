#!/bin/bash
#SBATCH --job-name=train_multiple
python main.py -dir stored_data/dyconv_effi_test38 | tee logs/dyconv_effi_test38.log
python main.py -dir stored_data/dyconv_effi_test39 | tee logs/dyconv_effi_test39.log
#python main.py -dir stored_data/dyconv_effi_test40 | tee logs/dyconv_effi_test40.log
