#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --account=project_462000999
#SBATCH --partition=small
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --array=0-15

# If run without sbatch, invoke here
if [ -z "$SLURM_JOB_ID" ]; then
	sbatch "$0" "$@"
    exit
fi

# See http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail

module purge
module use /appl/local/csc/modulefiles
module load pytorch
#source .venv/bin/activate

#[[ "$PYTHONPATH" != *"/scratch/project_462000353/tlundber/pythonuserbase/lib/python3.11/site-packages"* ]] && \
#export PYTHONPATH="/scratch/project_462000353/tlundber/pythonuserbase/lib/python3.11/site-packages:$PYTHONPATH"

pip install -r requirements.txt

read -r pca neighbors dist < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" param-grid-fasttext-embeddings.txt)

model="fastText"
data_name="SACX keywords"
#data="hplt"
#data="CORE"

echo "$model" "$data_name" "cluster metrics"

#: '
srun python clusters.py --data="$DATA/kw-embeddings/fasttext/centroid-norm" \
                           --langs="['en', 'fr', 'ur', 'zh']" \
                           --data_name="$data_name" \
                           --model_name="$model" \
                           --labels="all" \
                           --hover_text="text" \
                           --cmethod="all" \
						   --pca_before_umap="$pca" \
                           --rmethod="umap" \
                           --n_umap="[2,9,1]" \
                           --n_neighbors="$neighbors" \
                           --min_dist="$dist" \
                           --save_dir="$DATA/sacx-keyword-cluster-plots/fasttext/umap-grid/$pca-$neighbors-${dist//./}" \
						   --column_e="['embed_last']" \
						   --column_l="preds" \
# '

sacct -j "$SLURM_JOB_ID"
exit 0
