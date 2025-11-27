class ForecastingService:
    def __init__(
        self,
        *,
        xgb: XGBoostService | None = None,
        shap: ShapService | None = None,
        chroma_repo: ChromaRepository | None = None,
        db_repo: DBRepository | None = None,
        normalize: DataNormalizeService | None = None,
    ):
        self._xgb = xgb or XGBoostService()
        self._shap = shap or ShapService()
        self._chroma = chroma_repo or ChromaRepository()
        self._db = db_repo or DBRepository()
        self._norm = normalize or DataNormalizeService()

        ...