def explain_prediction(
    self, X: np.ndarray[Any, np.dtype[np.float64]], model: Any
) -> ShapExplanation:
    """
    Calcola i valori SHAP per le feature del modello.
    1. costruisce un TreeExplainer sul modello XGBoost;
    2. applica l’explainer alle righe di input;
    3. restituisce i valori SHAP incapsulati in ShapExplanation.
    """
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    shap_response = ShapExplanation(values=np.array(explanation.values))
    return shap_response
