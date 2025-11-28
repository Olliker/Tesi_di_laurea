from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import shap

from forecasting.app.models.shap_explanation import ShapExplanation


class ShapService:
    def explain_prediction(
        self, X: np.ndarray[Any, np.dtype[np.float64]], model: Any
    ) -> ShapExplanation:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X)
        shap_response = ShapExplanation(values=np.array(explanation.values))
        return shap_response

    def save_summary_plot(
        self,
        shap_explanation: ShapExplanation,
        *,
        features: np.ndarray[Any, np.dtype[np.float64]],
        feature_names: list[str],
        output_path: str | Path,
    ) -> Path | None:
        shap_values = np.asarray(shap_explanation.values, dtype=float)
        feats = np.asarray(features, dtype=float)

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)

        if shap_values.shape[0] != feats.shape[0]:
            raise ValueError(
                "SHAP values and feature matrix must have the same number of rows"
            )

        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        shap_values, feats = self._limit_plot_rows(shap_values, feats)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            feats,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path


    def generate_summary_plot(
        self,
        shap_explanation: ShapExplanation,
        *,
        features: np.ndarray[Any, np.dtype[np.float64]],
        feature_names: list[str],
        output_dir: Path,
    ) -> dict[str, str]:
        plots: dict[str, str] = {}

        summary_plot_path = output_dir / "shap_summary_plot.png"
        summary_path = self.save_summary_plot(
            shap_explanation,
            features=features,
            feature_names=feature_names,
            output_path=summary_plot_path,
        )
        if summary_path is not None:
            plots["summary_plot"] = str(summary_path)

        return plots

    @staticmethod
    def _limit_plot_rows(
        shap_values: np.ndarray[Any, np.dtype[np.float64]],
        features: np.ndarray[Any, np.dtype[np.float64]],
        max_rows: int = 1000,
    ) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
        """
        SHAP plots can be very slow with thousands of rows; keep a representative
        slice to avoid timeouts and warnings during forecast runs.
        """
        rows = shap_values.shape[0]
        if rows <= max_rows:
            return shap_values, features

        indices = np.linspace(0, rows - 1, num=max_rows, dtype=int)
        return shap_values[indices], features[indices]
