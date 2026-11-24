import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import (calculate_metrics, ks_statistic, plot_ks_figure,
                         plot_roc_figure, validation_ks_threshold)
from src.preprocess import handle_missing_values


class MitraPreparationAndMetricsTest(unittest.TestCase):
    def test_total_charges_whitespace_becomes_numeric_before_fill(self):
        frame = pd.DataFrame({"TotalCharges": ["42.5", " ", None],
                              "MonthlyCharges": [42.5, 10, 20]})
        train, val, test = handle_missing_values(frame, frame, frame)
        for result in (train, val, test):
            self.assertTrue(pd.api.types.is_numeric_dtype(result["TotalCharges"]))
            self.assertEqual(result["TotalCharges"].tolist(), [42.5, 0.0, 0.0])
        monthly, _, _ = handle_missing_values(frame, frame, frame, type="Monthly")
        self.assertEqual(monthly["TotalCharges"].tolist(), [42.5, 10.0, 20.0])
        self.assertEqual(frame.loc[1, "TotalCharges"], " ")

    def test_threshold_and_figures_use_scores_and_correct_tie_direction(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        threshold = validation_ks_threshold(labels, scores)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(calculate_metrics(labels, scores, threshold)["recall"], 0.5)
        self.assertEqual(ks_statistic(labels, scores), 0.5)
        for plot in (plot_ks_figure, plot_roc_figure):
            fig = plot(labels, scores)
            self.assertEqual(len(fig.axes), 1)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
