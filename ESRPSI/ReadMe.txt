{ENHANCING SPATIAL RESOLUTION AND PRESERVING SPECTRAL INTEGRITY: A BAYESIAN APPROACH TO HYPERSPECTRAL-MULTISPECTRAL IMAGE FUSION (ESRPSI)}
This code consists of four folders with following description
1-ESRPSI (This is the main folder and also the root folder for this code)
2-Algos (This folder contains two algorithms files. One for joint spectral dictionary and scaling factors leaenin matrix, and the other one for spectral sparse coefficients learning)
3-Utilities (This folder contains a file utils.py for images evaluation functions)
4-Results (The code execution will write the results in the respective files and will be saved in this folder)
5-Experiments (This folder contains the eperiments main functions)
6-Data (This folder contains datasets of our experimets  )
Please follow the following instrucitons to run the code
--Copy the folder ESRPSI to some location of the computer hard disk and take this folder as root to run the code.
--Install Python and the packages listed in the files in Experiments, Algos, and Utilities.
--If you run from the terminal, the current directory should be ESRPSI
Now Run the code as below from the terminal
Experiment1:
            type at the terminal: python Experiments/Exp1_senthesized_dataset_Fm_Fh.py
Experiment2:
            for part 1 type at the terminal: python exp2_averagingBlurr_part1.py
            for part 2 type at the terminal: python exp2_GaussianBlurr_part2.py
Experiment3:
            type at the terminal: python Experiments/Exp3_Isabella.py

Experiment4:
            for part 1 type at the terminal: python Exp4_tahoe_part1.py
            for part 2 type at the terminal: python Exp4_playa_part2.py
Note: Results will be shown at the teminal and will also be saved in seprate files in the folder "Results"
Wish you good luck