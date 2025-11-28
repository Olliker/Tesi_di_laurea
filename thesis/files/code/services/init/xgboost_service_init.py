class XGBoostService:
    
    def __init__(
        self,
        *,
        metrics: MetricsService | None = None,
        data_service: ForecastingDataService | None = None,
        config: XGBoostConfig | None = None,
    ):
        self.__cfg = config or XGBoostConfig()
        self.__data_service = data_service or ForecastingDataService()
        self.__metrics = metrics or MetricsService()
