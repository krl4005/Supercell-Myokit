#! /bin/bash -l

#SBATCH --partition=panda
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --job-name=supercellGA
#SBATCH --time=120:00:00             #HH/MM/SS
#SBATCH --mem=10G                  #memory requested, units available K,M,G,T

#source ~/ .bashrc

echo "Starting at:" `date` >> ga_analysis.txt
sleep 30
echo "This is job #:" $SLURM_JOB_ID >> ga_analysis.txt
echo "Running on node:" `hostname` >> ga_analysis.txt
echo "Running on cluster:" $SLURM_CLUSTER_NAME >> ga_analysis.txt
echo "This job was assigned the temporary (local) directory:" $TMPDIR >> ga_analysis.txt


module load python-3.7.6-gcc-8.2.0-hk56qj4
module load sundials/5.7.0
python3 data_analysis.py 