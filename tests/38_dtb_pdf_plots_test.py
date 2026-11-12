"""
Test 38: histograms of the learning database (LearningDatabase.plot_pdf_var and compare_resampled_pdfs).

The bin width is adapted to each variable (Freedman-Diaconis rule, utilities.smart_binwidth): a mass fraction
in [0, 0.1] gets many bins, not a single one as with a fixed width of 10.

Independent of the other tests.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_reacting_flows.databases_processing.database_processing import LearningDatabase


def database(folder):
    """LearningDatabase with only the attributes read by the histograms (no database read)."""

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"Temperature": rng.uniform(300.0, 2000.0, 1000), "H2O": rng.uniform(0.0, 0.1, 1000)})

    dtb = LearningDatabase.__new__(LearningDatabase)
    dtb.dtb_folder = str(folder)
    dtb.X = X
    dtb.X_old = X
    dtb.is_resampled = True
    dtb.is_pca_computed = False

    return dtb


def test_pdf_of_mass_fraction(tmp_path):
    dtb = database(tmp_path)

    # One histogram
    dtb.plot_pdf_var("H2O")
    assert len(plt.gca().patches) >= 10
    assert os.path.isfile(tmp_path / "distribution_H2O_resampled.png")
    plt.close("all")

    # Before and after re-sampling
    dtb.compare_resampled_pdfs("H2O")
    assert len(plt.gca().patches) >= 20
    assert os.path.isfile(tmp_path / "comparison_distribution.png")
    plt.close("all")
