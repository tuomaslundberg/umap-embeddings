#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --account=project_462000353
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

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

[[ "$PYTHONPATH" != *"/scratch/project_462000353/tlundber/umap-embeddings/pythonuserbase/lib/python3.10/site-packages"* ]] && \
export PYTHONPATH="/scratch/project_462000353/tlundber/umap-embeddings/pythonuserbase/lib/python3.10/site-packages:$PYTHONPATH"

pip install -r requirements.txt

model="bge-m3-fold-6"
data="cleaned"
#data="hplt"
#data="CORE"

echo $model $data "cluster metrics"

#: '
srun python clusters.py --data="/scratch/project_462000353/tlundber/umap-embeddings/data/model_embeds/${data}/${model}/" \
                           --langs="['en', 'fi', 'fr', 'sv']" \
                           --data_name=$data \
                           --model_name=$model \
                           --labels="all" \
                           --hover_text="text" \
                           --sample=2400 \
                           --cmethod="all" \
                           --rmethod="umap" \
                           --n_umap="[2,4,1]" \
                           --save_dir="/scratch/project_462000353/tlundber/umap-embeddings/data/cluster_plots/${data}/${model}/with_hover/" \
						   --column_e="['embed_last', 'embed_first']" \
						   --column_l="preds" \
						   --keep_sublabels="True" \
# '

sacct -j "$SLURM_JOB_ID"
exit 0
