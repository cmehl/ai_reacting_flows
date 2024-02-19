import pickle
import glob

from tensorflow.keras.models import model_from_json
from ai_reacting_flows.ann_model_generation.tensorflow_custom import GetN2Layer, ZerosLayer, GetLeftPartLayer, GetRightPartLayer
from ai_reacting_flows.ann_model_generation.tensorflow_custom import ResidualBlock


path_model_pytorch = "./Convert_Pytroch_pkl"
path_model_tensorflow = "./Tensorflow/MODEL_Learning_Tensorflow"


nb_clusters = len(glob.glob1(path_model_tensorflow,"*.json"))
print(f">> {nb_clusters} models were found")

for ite in range(0,nb_clusters): 
    #LOADS PYTORCH WEIGHTS

    with open(path_model_pytorch + f"/model_weight_cluster_{ite}_pytorch.pkl",'rb') as f :
        weight_pytorch = pickle.load(f)


    #START TENSORFLOW MODEL
    with open(path_model_tensorflow+ f"/model_architecture_cluster{ite}.json",'r') as f : 
        model = model_from_json(f.read(), custom_objects={'GetN2Layer' : GetN2Layer,
                                                                            'ZerosLayer': ZerosLayer,
                                                                            'GetLeftPartLayer': GetLeftPartLayer,
                                                                            'GetRightPartLayer': GetRightPartLayer,
                                                                            'ResidualBlock': ResidualBlock})
    #LOAD TENSORFLOW WEIGHTS 
    model.load_weights(path_model_tensorflow+f"/model_weights_cluster{ite}.h5")

    layers_tensorflow = []
    for layer in model.layers : 
        #print(layer.name)
        layers_tensorflow.append(layer.name)

    layers_pytorch = [] 
    for names, values in weight_pytorch.items() : 
        #print(names)
        layers_pytorch.append(names)
    i = 0 
    for layers_int in layers_tensorflow[1:] : # On enlève le input layers
        weight = weight_pytorch[layers_pytorch[i]].T
        bias = weight_pytorch[layers_pytorch[i+1]].T

        layer = model.get_layer(layers_int)
        layer.set_weights([weight,bias])

        i=i+2 #Weiht and bias are paires (N and N+1) so to get to the next layer you put i= i+2 


    model.save_weights(f"./Convert_pkl_Tensorflow/Model_pytorch_{ite}_converti_tensorflow.h5")
