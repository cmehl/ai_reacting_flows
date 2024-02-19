import torch 
import pickle
import glob
import os 

path_model_pytorch = "./Pytorch/Learning_Pytorch"

nb_clusters = len(glob.glob1(path_model_pytorch,"*.pt"))
print(f">> {nb_clusters} models were found")


for g in range(0,nb_clusters):

    weight = torch.load(path_model_pytorch + f'/Model_{g}_Pytorch',map_location=torch.device('cpu'))

    keys = []
    for names,values in weight.items() : 
        
        keys.append(names)
        
    print(keys)

    dico = {}
    for i in keys : 
        dico[i] = weight[i].cpu().detach().numpy()

    #print(dico)

    with open(f"./Convert_Pytorch_pkl/model_weight_cluster_{g}_pytorch.pkl","wb") as f :
        pickle.dump(dico,f)
