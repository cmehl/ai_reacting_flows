# PK3 — pipeline ARF entraîné sur la base CFD (rollout 3, dt 5e-6)

Catalogue des scripts les plus importants pour le modèle **PK3** (PK3-H / PK3-noH dans la présentation
`presentation_figs_2`), c'est-à-dire la voie **ARF entraînée sur la base CFD** (81.2 M états LES, rollout 3 pas
Cantera à dt=5e-6, k-means 3, per-species 2x64 tanh) — PAR OPPOSITION à la voie entraînée sur réacteurs
stochastiques (STOCH1/2/5/6/7, pipeline séparé, non couvert ici).

**Rien n'a été supprimé** : ceci est une copie additive des scripts jugés les plus importants, faite pour
documenter le pipeline avant le nettoyage d'orion / du disque dur externe. Les copies ci-dessous sont figées à la
date du 2026-10-29 ; les originaux (avec tous les logs, poids entraînés, bases HDF5, CSV de résultats) restent sur
orion à `/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea3/MLP_CFD_TFM_AMR_ROLLOUT3/`.
Les calculs CONVERGE bruts (SAGE, PK3-H, PK3-noH, ...) sont en cours de copie vers le disque dur externe
`/run/media/kotlarcm/9E8A56E08A56B48D/MARTIN/CONVERGE/`.

## 1. Vue d'ensemble du pipeline

```
   base LES/CFD (81.2 M états, T > 800 K, échantillonnage uniforme en T)
        │
        ▼  preselect_T1D.py           (reproduit la sélection mc50000 sans recharger les 271 Go)
   selected_T1D_mc50000.npz
        │
        ▼  build_rollout_from_raw.py  (MPI, 3 pas Cantera à dt=5e-6 par état sélectionné)
   solutions_rollout.h5
        │
        ▼  dtb_processing_rollout.py  (wrapper LearningDatabase : seuil T, k-means 3, split train/val)
   dtb_rollout3_thresh1e11_kmeans3_T800_mc50000/   (train/val CSV + h5 par cluster)
        │
        ▼  ann_model_learning_weighted.py → nn_manager_weighted.py   (entraînement, 1 job par cluster z0/z1/z2)
   MODELS/MODEL_TFM_AMR_ROLLOUT3_dt5e6_persp_kmeans3roll_dt5e6/   (poids .pth + normalisations)
        │
        ▼  test_full_domain.py        (a priori, MPI, vs CVODE sur post107 — PAS encore CONVERGE)
   results/parts_*/full_domain_*.h5
        │
        ▼  déploiement dans CONVERGE (TFM+AMR, dt 5e-6, binaire converge_accel, renorm_yk=3)
   CAS_TFM_AMR_PERSP_K3ROLL_DT5E6_TOL3E5 (PK3-H)  /  _NOHYB (PK3-noH)
        │
        ▼  scripts converge_postprocessing/ (ce dossier)
   recap, erreurs par espèce, champs, split de chaleur CVODE/réseau, seuil hybride, diagnostic de crash
```

L'orchestration complète de ce chemin (build → processing → split k-means → 3 entraînements → assemblage →
tests a priori → merges) est scriptée en un seul fichier : `pipeline/setup_k3_dt5e6.sh`, qui génère et soumet
toute la chaîne SLURM via `sbatch --dependency=afterok:...` (plus fiable qu'un watcher `nohup` en arrière-plan,
qui meurt quand la session SSH se ferme — orion n'a pas `loginctl enable-linger`).

## 2. `pipeline/` — construction de la base et entraînement

| Script | Rôle |
|---|---|
| `preselect_T1D.py` | Rejoue l'algorithme d'undersampling en température (`LearningDatabase.undersample_1D`, min_count=50000) sur la base brute CFD (81 222 200 lignes) en streaming, sans charger les 271 Go en mémoire. Écrit les indices globaux sélectionnés (`selected_T1D_mc50000.npz`). |
| `build_rollout_from_raw.py` | MPI (`mpirun`). Pour chaque état sélectionné, fait avancer un réacteur à pression constante (Cantera) de 3 pas de dt=5e-6 (rollout 3), écrit X/Y/Y_multi. Fusionné par le rang 0 en `solutions_rollout.h5`. |
| `dtb_processing_rollout.py` | Fin wrapper autour de `ai_reacting_flows.databases_processing.database_processing.LearningDatabase` : seuil de température, clustering k-means 3, écriture train/val par cluster. Piloté par `dtb_processing.yaml`. |
| `dtb_processing.yaml` | Config type pour `dtb_processing_rollout.py` (suffixe de dossier, fichier mécanisme, nb_clusters). |
| `ann_model_learning_weighted.py` | Point d'entrée d'entraînement (`python ann_model_learning_weighted.py <train_folder>`), délègue à `NNManagerWeighted`. |
| `nn_manager_weighted.py` | Variante de `NN_manager` (src ARF) avec perte MSE pondérée par l'inverse de l'incrément (`loss_weighting: inv_increment`, étude du biais gaz brûlés) et pénalité optionnelle de conservation des éléments (`element_penalty`). C'est la version utilisée pour entraîner PK3 — remplace `nn_manager_weighted_before_elem_20261008.py` (ancienne version, restée sur orion uniquement, ne pas réutiliser). |
| `test_full_domain.py` | Test a priori MPI (`FullDomainTester` du src ARF) : réseau vs CVODE sur l'ensemble du domaine chaud d'un post CONVERGE (ex. post107), piloté par `test_ann_vs_cvode.yaml`. C'est le test *avant* tout lancement CONVERGE. |
| `setup_k3_dt5e6.sh` | Orchestration complète de la chaîne ci-dessus (génère les `.sbatch`, les soumet avec les bonnes dépendances). C'est le script qui a produit PK3-H / PK3-noH. |

## 3. `converge_postprocessing/` — validation dans CONVERGE (PK3-H / PK3-noH vs SAGE)

Tous ces scripts lisent les sorties CONVERGE brutes (`thermo.out`, `species_mass.out`, `converge.log`, les posts
HDF5) sans dépendance aux CSV lourds de post-traitement CONVERGE — le même patron est réutilisé partout
(fonction `segments()`/`read_out()` qui recolle `outputs_original` + `outputs_restartN`).

| Script | Rôle | Slide(s) associée(s) |
|---|---|---|
| `recap_pk3.py` | Tableau récapitulatif complet (erreur masse moyenne 18 espèces, ΔT, erreur HRR, % appels CVODE, chaleur produite par CVODE, gain de vitesse sagetime, écarts éléments O/H/N) pour PK3-H / PK3-noH (+ STOCH5 pour référence), sur toute la durée du run. | `recap-pk3` |
| `species_profiles_3cas.py` | 18 sous-graphes (un par espèce), fraction massique moyenne du domaine vs temps, PK3-H/PK3-noH (+ STOCH5), échelle log. Modèle réutilisé pour `species_profiles_pk3.py` et `species_profiles_stoch67.py`. | (base de `species-pk3`) |
| `species_profiles_pk3.py` | Même chose, restreint à PK3-H/PK3-noH seuls, fenêtre complète 0–100 ms. | `species-pk3` (ajoutée le 2026-10-29) |
| `q_chem_split.py` | Reconstruction offline (Cantera) du dégagement de chaleur produit par CVODE vs par le réseau, cellule par cellule, car `MASS_SOURCE` dans les posts n'est pas une source chimique pure. Vérifié contre `thermo.out` (ratio ≈ 1.04). Job array SLURM. | `hybride-front` |
| `champs_4cas.py` | Champs 2D (T + 2 espèces majoritaires + pire espèce) sur une coupe X=0, PK3-H/PK3-noH/STOCH5/STOCH6, au même instant, vs SAGE. | `condt-pk3`, `elements-pk3` |
| `apriori_step.py` + `apriori_merge.py` | Calibration a priori du seuil hybride ANN/CVODE (méthode Mehl & Aubagnac-Karkar, *Proc. Combust. Inst.* 2023, Fig. 11-12) : comparaison du critère CONVERGE (dérive max relative des éléments sur la sortie brute du réseau) à l'erreur réelle vs CVODE, sur tout le domaine chaud de SAGE à plusieurs instants. Job array + fusion. | `seuil-hybride` |
| `hybrid_slide.py` | Figure de synthèse du balayage de seuil (coût en % de cellules/temps CVODE vs erreur résiduelle). | `seuil-hybride` |
| `mlp_crash_cells.py` | Diagnostic du crash du MLP 2x256 K3-ROLL sans hybride à t=4.195 ms (poche d'azote à l'injecteur) : localisation des cellules à saut de température non physique, comparaison avec PK3-noH et SAGE au même instant. | `crash-mlp` |

## 4. Ce qui N'EST PAS dans ce dossier (volontairement)

- Le pipeline STOCH1/2/5/6/7 (réacteurs stochastiques + flammelettes + relaxation) — voie B de la présentation,
  sujet distinct, scripts dans les mêmes dossiers orion mais préfixés `*_stoch*`.
- Les bases de données brutes/traitées (`.h5`, plusieurs centaines de Go au total), les poids entraînés (`.pth`),
  les CSV de résultats bruts — tout reste sur orion (`.idea3/MLP_CFD_TFM_AMR_ROLLOUT3/`) et/ou sur le disque dur
  externe pour les runs CONVERGE. Ce dossier ne contient que le *code*.
- `nn_manager_weighted_before_elem_20261008.py` : ancienne version de `nn_manager_weighted.py` avant l'ajout de
  la pénalité de conservation des éléments — gardée sur orion par précaution mais non recopiée ici (superseded).
- `submit_tfm_rollout3.sh`, `launch_all_20261009.sh` : variantes/essais d'orchestration antérieurs à
  `setup_k3_dt5e6.sh` (3-zone T-split, perte pondérée seule, etc.) — pas la chaîne qui a produit PK3-H/PK3-noH
  tels que présentés dans le deck. Laissés sur orion, non recopiés.
- Les scripts d'analyse exploratoire secondaires (`check_n2h2_zoom.py`, `pocket_history.py`,
  `slices_YN_pocket*.py`, `ythr_*.py`, `qcvode_proxy.py`, `recap_4cas.py` — ce dernier rejeté par l'utilisateur
  en faveur de `species_profiles_pk3.py`) : utiles pour l'historique mais pas le cœur du pipeline.

## 5. Dépendances

Tous ces scripts s'appuient sur le package installable `src/ai_reacting_flows/` de ce repo (notamment
`databases_processing.database_processing.LearningDatabase`, `ann_model_generation.cfd_full_domain_testing.FullDomainTester`,
`ann_model_generation.NN_manager`) — voir le `CLAUDE.md` racine du repo pour l'environnement (`ml_arf` +
`.venv`) et l'installation (`pip install -e .`). Les scripts MPI (`build_rollout_from_raw.py`, `test_full_domain.py`,
`q_chem_split.py`, `apriori_step.py`) nécessitent OpenMPI/4.1.1 et sont lancés via `mpirun`/SLURM, pas en direct.
