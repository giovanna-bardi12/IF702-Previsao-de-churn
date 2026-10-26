#!/bin/bash
#SBATCH --job-name=churn-kan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -p short-simple
#SBATCH --qos=simple
#SBATCH --output=slurm_%j.out
#SBATCH --error=job_error_%j.txt
set -e

PROJECT=IF702-Previsao-de-churn
VENV=~/churn_venv

cdir=${TMPDIR}/churn${SLURM_JOB_ID}
mkdir -p "$cdir"
chmod 700 "$cdir"
cd "$cdir"

function clean_up {
    echo ">> Sincronizando resultados de volta para a home e limpando o node..."
    sync_src="$cdir/$PROJECT"
    if [ -d "$sync_src" ]; then
        # Traz de volta: churn_optuna.db, wandb/ (checkpoints), test_results/, data/data_processed/
        rsync -av --exclude='*__pycache__*' --exclude='slurm*' --exclude='.git' --exclude='venv' \
            "$sync_src/" ~/"$PROJECT"/
    else
        echo "Warning: '$sync_src' não existe, pulando rsync."
    fi
    cd ..
    rm -rf "$cdir"
    echo ">> Limpeza concluída."
}


trap 'clean_up' EXIT SIGTERM

module purge
module use /opt/easybuild/modules/all
module load Python/3.10.8-GCCcore-12.2.0

if [ ! -d ~/"$PROJECT" ]; then
    echo "Pasta ~/$PROJECT não encontrada. Saindo."
    exit 1
fi
if [ ! -f ~/"$PROJECT"/data/data_raw/customer_churn_telecom_services.csv ]; then
    echo "Dataset não encontrado em ~/$PROJECT/data/data_raw/. Saindo."
    exit 1
fi

# Copia o projeto para o disco local do node (sem o venv local nem o .git)
rsync -av --exclude='.nfs*' --exclude='venv' --exclude='.git' ~/"$PROJECT" .
cd "$PROJECT"

# O venv é criado uma vez no nó de login (junto com o "wandb login"), não dentro do job
if [ ! -d "$VENV" ]; then
    echo "Venv $VENV não encontrado. Crie no nó de login antes de submeter:"
    echo "  module load Python/3.10.8-GCCcore-12.2.0 && python3 -m venv $VENV"
    echo "  source $VENV/bin/activate && pip install -r requirements.txt && wandb login"
    exit 1
fi
if [ ! -f ~/.netrc ] && [ -z "$WANDB_API_KEY" ]; then
    echo "Aviso: login do W&B não encontrado (~/.netrc). Rode 'wandb login' no nó de login."
fi
 

source "$VENV"/bin/activate

export MPLBACKEND=Agg          # matplotlib sem display
export PYTHONUNBUFFERED=1      # prints aparecem na hora no log do slurm

nvidia-smi || echo "Sem GPU visível, rodando em CPU."
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

# -------- pré-processamento
#python -m src.prepare_data

if [ "$STAGE" = "baseline" ]; then
    # -------- baseline: treino com hiperparâmetros fixos + teste
    python kan/kan_baseline.py
    python kan/kan_test.py --baseline
else
    # -------- treino + validação (Optuna) + teste das 5 melhores trials
    python kan/kan_train.py
    python kan/kan_test.py --top-k 5
fi
 
echo "Finish"