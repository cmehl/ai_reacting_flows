HOW TO USE PYTORCH LEARNING AND TESTING PART 1 

Install the enviro_pytorch.txt : pip install -r enviro_pytorch.txt 
(same as the enviro_tensorflow with the pytorch framework). No needed to load cuDNN (pytoch install one itself)

!!! Need to seperated venv one for pytorch the other for tensorflow !!!

IF PYOTRCH NOT WORKING TRY : 
pip install torch torchvision torchaudio 

or 

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

The file ann_model_learning_pytorch work like the ann_model_learning from tensorflow. 
the same informations are needed , juste separated in communs, training and testing 

scripts can be used to initiate learning on a ENER440 

Same for the ann_model_testing_pytorch. Same architecutre as the learning. 

don't forget to change the path of the stochastic database/mech_file etc.

CONVERT PYTORCH TO TENSORFLOW : 

Convert_Pytorch_pkl.py : (with venv pytorch)

Change the Model Path of pytorch model 


Convert_pkl_tensorflow.py : (with venv tensorflow)

change the model path of tensorflow model 

after the convert , you have to set the testing parameters "output_weight" of tensorflow to true. 


