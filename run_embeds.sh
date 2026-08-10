#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --account=project_2005092
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# If run without sbatch, invoke here
if [ -z "$SLURM_JOB_ID" ]; then
    sbatch "$0" "$@"
    exit
fi

# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

data_name="cleaned" #$1
model_name="bge-m3" #$2
fold=6 #$3

case $data_name in
    CORE)
        langs=("en" "fi" "sv") #("en" "fi" "fr" "sv" "tr")
        ;;
    register_oscar)
        langs=("en" "fr" "ur" "zh")
        ;;
    hplt)
        langs=("en" "fr" "ur" "zh")
        ;;
    balanced_register_oscar)
        langs=("en" "fr" "ur" "zh")
        ;;
    cleaned)
        langs=("en" "fi" "fr" "sv") #("en" "fa" "fi" "fr" "sv" "ur" "tr" "zh")
        ;;
    dirty)
        langs=("en" "fa" "fi" "fr" "sv" "ur" "tr" "zh")
        ;;
    *)
        echo "Invalid data_name. Please specify CORE or REG."
        exit 1
        ;;
esac

export HF_DATASETS_CACHE="/scratch/project_2005092/tlundber/hf_cache"

echo $langs, $data_name, $model_name, $fold
module purge
#module load LUMI
#module load PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315
module use /appl/local/csc/modulefiles
module load pytorch/2.4
which python3
for lang in "${langs[@]}"; do
	#srun python3 --version
	srun /appl/soft/ai/wrap/pytorch-2.6-new/bin/python3 --version
	#srun /appl/soft/ai/wrap/pytorch-2.6-new/bin/python3 embeds.py --lang=$lang --data_name=$data_name --model_name=$model_name --fold=$fold
    #echo "python3 embeds.py --lang=$lang --data_name=$data_name --model_name=$model_name --fold=$fold"
done
sacct --format="jobid,Elapsed" -j $SLURM_JOBID 
mkdir -p logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
mv logs/${SLURM_JOBID}.* logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
exit 0
