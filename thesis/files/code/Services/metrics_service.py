class MetricsService:
    def mape_percent(
        self, y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7
    ) -> float:
        denom = np.where(np.abs(y_true) < eps, eps, y_true)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    def smape_percent(
        self, y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7
    ) -> float:
        denom = np.maximum(eps, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
        return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

    def wape_percent(
        self, y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7
    ) -> float:
        den = max(eps, float(np.sum(np.abs(y_true))))
        return float(np.sum(np.abs(y_true - y_pred)) / den * 100.0)

