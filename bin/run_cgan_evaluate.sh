#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=200gb
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cgan-eval
#SBATCH --partition cnu
#SBATCH --output=logs/cgan-eval-%A.out

# try with   --gres=gpu:A100:1
source ~/.bashrc
source ~/.initConda.sh
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

# checkout what the gpus are doing
nvidia-smi --query-gpu=gpu_name,driver_version,memory.free,memory.total --format=csv

nvidia-smi
module load lang/cuda/11.2-cudnn-8.1
nvidia-smi

# print out stuff to tell you when the script is running
echo running model
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

# run the script to train your model
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7ed5693482c955aa_small-cl1000 --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7ed5693482c955aa_small-cl100-weightedclossclipped --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7ed5693482c955aa_small-cl500-weightedclossclipped --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7ed5693482c955aa_small-cl2000 --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7ed5693482c955aa_small-cl1000-modifiedlr --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7ed5693482c955aa_cl1000-with-cin --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/76b8618700c90131_cl10-no-logs --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7ed5693482c955aa_cl1000-rotate-data  --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/76b8618700c90131_medium-cl20-no-logs --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/76b8618700c90131_medium-cl10-no-logs --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/76b8618700c90131_cl50-no-logs --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7ed5693482c955aa_small-2step-ds --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/5c577a485fbd1a72_0876a13533d2542c --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7ed5693482c955aa_medium-cl2000 --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/db6efa12ff2540b4_ifs-qmap --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7ed5693482c955aa_medium-cl2000 --num-images 4000 --eval-ensemble-size 10 --val-ym-start 201806 --val-ym-end 201905;
# srun python -m dsrnngan.main --no-train --eval-short --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/76b8618700c90131_cl100-no-logs --num-images 1000 --eval-ensemble-size 20 --val-ym-start 201903 --val-ym-end 201905;

# srun python -m dsrnngan.main --no-train --eval-model-numbers 52800 --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/a4822ed35a44475d_small-cl100-nologs-nocrop --num-images 2900 --eval-ensemble-size 1 --save-generated-samples;
# srun python -m dsrnngan.main --no-train --eval-model-numbers 62400 --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/a4822ed35a44475d_small-cl50-nologs-nocrop --num-images 2900 --eval-ensemble-size 1 --save-generated-samples;

# srun python -m dsrnngan.main --no-train --save-generated-samples --eval-model-numbers 230400 --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs --num-images 2900 --eval-ensemble-size 20;

# srun python -m dsrnngan.main --no-train --eval-model-numbers 268800 --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/998e5f0f54106b35_medium-cl50-nologs-nocrop-seplake --num-images 2900 --eval-ensemble-size 1;
# srun python -m dsrnngan.main --no-train --save-generated-samples --eval-model-numbers 230400 --no-shuffle-eval --model-folder /user/work/uz22147/logs/cgan/ec66788090f69890_small-cl50-lvonly --num-images 2900 --eval-ensemble-size 1;

srun python -m dsrnngan.main --no-train  --eval-model-numbers 217600 --model-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs --num-samples 10 --eval-ensemble-size 1 --val-ym-start 202010 --val-ym-end 202011;

# srun python -m dsrnngan.main --no-train --no-shuffle-eval --eval-model-numbers 217600 --save-generated-samples --model-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs --num-images 2088 --eval-ensemble-size 1 --val-ym-start 201803 --val-ym-end 201805;

# srun python -m dsrnngan.main --no-train --eval-model-numbers 217600 --save-generated-samples --model-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs --num-images 18000 --eval-ensemble-size 1 --eval-on-train-set;


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

