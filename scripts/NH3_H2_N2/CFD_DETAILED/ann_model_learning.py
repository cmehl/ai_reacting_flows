"""Train one of the three NH3_H2_N2 architecture variants for this case.

Copies the chosen networks_params_<variant>.yaml over the fixed-filename
networks_params.yaml that NN_manager reads (NN_manager always reads exactly
"networks_params.yaml" in the run folder -- see
ai_reacting_flows.ann_model_generation.NN_manager.NN_manager.__init__),
then trains. Output model lands in MODELS/MODEL_<model_name_suffix> (see
the chosen yaml's model_name_suffix).

Usage:
    python ann_model_learning.py <mlp_percluster|perspecies|resnet>
"""
import sys
import shutil

from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager

VARIANTS = ["mlp_percluster", "perspecies", "resnet"]

if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
    raise SystemExit(f"Usage: python {sys.argv[0]} <{'|'.join(VARIANTS)}>")

variant = sys.argv[1]
shutil.copy(f"networks_params_{variant}.yaml", "networks_params.yaml")

mlp_model = NN_manager()
mlp_model.train_all_clusters()
