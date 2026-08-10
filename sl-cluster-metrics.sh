#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --partition=small
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-logs/array_%A_%a.out
#SBATCH --error=slurm-logs/array_%A_%a.err
#SBATCH --array=1-8
source ~/lumi-env.sh

# If run without sbatch, invoke here
if [ -z "$SLURM_JOB_ID" ]; then
	sbatch "$0" "$@"
    exit
fi

# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

# Make logging a bit easier
mkdir -p slurm-logs
rm -f "slurm-logs/current_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.err"
rm -f "slurm-logs/current_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.out"
ln -s "array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "slurm-logs/current_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.err"
ln -s "array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "slurm-logs/current_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.out"

# Load the Python environment
module purge > /dev/null 2>&1 # Get rid of boilerplate stderr
module use /appl/local/csc/modulefiles > /dev/null 2>&1
module load pytorch > /dev/null 2>&1

#pip install -r requirements.txt

registers=("MT" "LY" "SP" "ID" "NA" "HI" "IN" "OP" "IP")
reg=${registers[$SLURM_ARRAY_TASK_ID]}

model="sentence-transformers/LaBSE"
data_name="SACX keywords, register $reg"
#data="hplt"
#data="CORE"

echo "$model" "$data_name" "$reg" "cluster metrics"

#: '
python clusters.py --data="$DATA/kw-embeddings/final" \
                           --langs="['en', 'fr', 'ur', 'zh']" \
                           --data_name="$data_name" \
                           --model_name="$model" \
                           --labels="['$reg']" \
                           --hover_text="['text', 'script_type', 'translation', 'comments']" \
                           --truncate_hover=False \
						   --pca_before_umap="50" \
						   --n_neighbors="8" \
                           --cmethod="spherical-kmeans" \
                           --rmethod="umap" \
                           --n_umap="[2,10,1]" \
                           --seed=42 \
                           --save_dir="$DATA/sacx-keyword-cluster-plots/final/wordcloud" \
						   --column_e="['embed_last']" \
						   --column_l="preds" \
# '

# Walltime statistics
sacct -o jobid,elapsed -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

exit 0
