#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=200gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=plots
#SBATCH --partition shared
#SBATCH --output=logs/plots-%A.out

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

# print out stuff to tell you when the script is running
echo running plots for cgan
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

srun python -m scripts.make_plots --nickname cl100-medium-nologs-fullyear --model-eval-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs/n8640_202010-202109_45682_e1 --model-number 217600 --output-dir /user/home/uz22147/repos/downscaling-cgan/plots -fss --climatological-data-path /bp1/geog-tropical/users/uz22147/east_africa_data/daily_rainfall;

# srun python -m scripts.make_plots --nickname cl100-small-nologs --log-folder /user/work/uz22147/logs/cgan/a4822ed35a44475d_small-cl100-nologs/n2900_201806-201903_42e34_e1 --model-number 57600 --diurnal -ex -qq -rapsd -hist -csi --output-dir /user/home/uz22147/repos/downscaling-cgan/plots
# srun python -m scripts.make_plots --nickname cl100-small-nologs-nocrop --log-folder /user/work/uz22147/logs/cgan/a4822ed35a44475d_small-cl100-nologs-nocrop/n2900_201806-201903_42e34_e1 --model-number 52800 --diurnal -ex -qq -rapsd -hist -csi --output-dir /user/home/uz22147/repos/downscaling-cgan/plots
# srun python -m scripts.make_plots --nickname cl200-small-nologs-nocrop --log-folder /user/work/uz22147/logs/cgan/a4822ed35a44475d_small-cl200-nologs-nocrop/n2900_201806-201903_42e34_e1 --model-number 67200 --diurnal -ex -qq -rapsd -hist -csi --output-dir /user/home/uz22147/repos/downscaling-cgan/plots
# srun python -m scripts.make_plots --nickname cl20-small-nologs-nocrop --log-folder /user/work/uz22147/logs/cgan/a4822ed35a44475d_small-cl20-nologs-nocrop/n2900_201806-201903_42e34_e1 --model-number 57600 --diurnal -hist -csi --output-dir /user/home/uz22147/repos/downscaling-cgan/plots

# srun python -m scripts.make_plots --nickname cl2000-medium --log-folder /user/work/uz22147/logs/cgan/457221781d3b7cc9_medium-cl2000-final/n2900_201806-201903_42e34_e20 --model-number 262400 --diurnal -ex -qq -rapsd -hist -csi --output-dir /user/home/uz22147/repos/downscaling-cgan/plots
# srun python -m scripts.make_plots --nickname cl100-medium-nologs --log-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs/n8640_202010-202109_45682_e1 --model-number 217600 --output-dir /user/home/uz22147/repos/downscaling-cgan/plots  --diurnal -ex -qq -rapsd -hist -csi -fss
# srun python -m scripts.make_plots --nickname cl100-medium-nologs --log-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs/n1000_202010-202109_45682_e100 --model-number 217600 --output-dir /user/home/uz22147/repos/downscaling-cgan/plots -rh 
# srun python -m scripts.make_plots --nickname cl100-medium-nologs --log-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs/n2088_201803-201805_f37bd_e1 --model-number 217600 --output-dir /user/home/uz22147/repos/downscaling-cgan/plots --diurnal -ex -qq -rapsd -hist -csi -fss

# srun python -m scripts.make_plots --nickname cl5-lv-only --log-folder /user/work/uz22147/logs/cgan/ec66788090f69890_small-cl5-lvonly/n2900_201806-201903_42e34_e1 --model-number 288000 --output-dir /user/home/uz22147/repos/downscaling-cgan/plots -fss
# srun python -m scripts.make_plots --nickname cl50-lv-only --log-folder /user/work/uz22147/logs/cgan/ec66788090f69890_small-cl50-lvonly/n2900_201806-201903_42e34_e1 --model-number 230400 --output-dir /user/home/uz22147/repos/downscaling-cgan/plots -fss
# srun python -m scripts.make_plots --nickname cl0.55-lv-only --log-folder /user/work/uz22147/logs/cgan/ec66788090f69890_small-cl0.5-lvonly/n2900_201806-201903_42e34_e1 --model-number 281600 --output-dir /user/home/uz22147/repos/downscaling-cgan/plots -fss


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
