#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1       # or a100:1, a5000:1, dgx:1, t4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4            # adjust (e.g., 8 or 16 if needed)
#SBATCH --mem=32G                    # memory per node
#SBATCH --time=5:00:00               # job time limit (HH:MM:SS)
#SBATCH --mail-user=jamal73sm@gmail.com
#SBATCH --mail-type=ALL


#cd /home/shahab33/projects/def-arashmoh/shahab33/GenFKD #Cedar

cd /project/def-arashmoh/shahab33/Rohollah/projects/FeD2P #Graham

#cd /project/def-arashmoh/shahab33/GenFKD #Narval 


module purge
module load python
module load cuda


#source /home/shahab33/FeDK2P/bin/activate  	# Cedar

source /home/shahab33/fed2p/bin/activate #Graham

#source /home/shahab33/fed2p/bin/activate #Narval

#python main1.py --local_model_name "ResNet20" --num_train_samples 10000 --alpha_dirichlet 10 --output_name "ResNet20_10K_alpha10_"  	# this is the direction and the name of your code
#python main.py --local_model_name "ResNet18" --num_train_samples 50000 --alpha_dirichlet 100 --output_name "ResNet18_50K_alpha100_"

python ImgGen.py
