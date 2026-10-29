# Usage: python ann_model_learning_weighted.py <train_folder>   (loss_weighting in its networks_params.yaml)
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_manager_weighted import NNManagerWeighted

NNManagerWeighted(run_folder=sys.argv[1]).train_all_clusters()
