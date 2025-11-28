def generate_summary_plot(
    self,
    shap_explanation: ShapExplanation,
    *,
    features: np.ndarray[Any, np.dtype[np.float64]],
    feature_names: list[str],
    output_dir: Path,
) -> dict[str, str]:
    """
    Genera il summary plot SHAP e restituisce il path del file salvato.
    """
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
