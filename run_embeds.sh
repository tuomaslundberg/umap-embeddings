#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --account=project_462000353
#SBATCH --time=72:00:00
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --array=0-4

# If run without sbatch, invoke here
if [ -z "$SLURM_JOB_ID" ]; then
	sbatch "$0" "$@"
    exit
fi

# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

data_name="concat" #$1
model_name="xlm-r-reference" #$2
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
	concat)
		langs=("en" "fi" "fr" "sv" "th")
		;;
    dirty)
        langs=("en" "fa" "fi" "fr" "sv" "ur" "tr" "zh")
        ;;
    *)
        echo "Invalid data_name. Please specify CORE or REG."
        exit 1
        ;;
esac

export HF_DATASETS_CACHE="/scratch/project_462000353/tlundber/hf_cache"

echo $langs, $data_name, $model_name, $fold
module purge
#module load LUMI
#module load PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315
module use /appl/local/csc/modulefiles
module load pytorch #/2.4

lang=${langs[$SLURM_ARRAY_TASK_ID]}

python -u embeds.py --lang=$lang --data_name=$data_name --model_name=$model_name
#echo "python3 embeds.py --lang=$lang --data_name=$data_name --model_name=$model_name"
sacct --format="jobid,Elapsed" -j $SLURM_JOBID
mkdir -p logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
mv logs/${SLURM_JOBID}.* logs/embeds_${model_name}_${fold}_${data_name}/${lang}/
exit 0
