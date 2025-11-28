class ForecastingDataService:
    """Versione ridotta per la tesi: passaggi principali di preparazione dati."""

    def __init__(
        self,
        *,
        data_frame_service: DataFrameService | None = None,
        invoice_service: InvoiceConversionService | None = None,
    ):
        self._dfs = data_frame_service or DataFrameService()
        self._invoice_service = invoice_service or InvoiceConversionService()

    def get_data_frame_service(self) -> DataFrameService:
        return self._dfs

    def get_price_columns(self, include_derivatives: bool = False) -> list[str]:
        return self._dfs.get_price_columns(include_derivatives=include_derivatives)

    def load_fact_df(self, cutoff_date: pd.Timestamp | None = None) -> pd.DataFrame:
        df = self._invoice_service.convert_db_data_to_df()
        date_col = self._dfs.get_date_col()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if cutoff_date is not None:
            df = df[df[date_col] < pd.to_datetime(cutoff_date).tz_localize(None)]
        return df

    def prepare_regression_frame(
        self, cutoff_date: pd.Timestamp | None = None
    ) -> tuple[pd.DataFrame, list[str], str, str]:
        raw = self.load_fact_df(cutoff_date=cutoff_date)
        date_col = self._dfs.get_date_col()
        target_col = self._dfs.get_target_col()
        prepared = self._dfs.prepare_features(raw).sort_values(date_col)
        feature_cols = self._dfs.get_feature_columns(prepared)
        return prepared, feature_cols, date_col, target_col

    def prepare_dataset(
        self, cutoff_date: pd.Timestamp | None = None
    ) -> tuple[pd.DataFrame, list[str], str, str]:
        return self.prepare_regression_frame(cutoff_date=cutoff_date)

    def build_future_rows(
        self,
        hist_df: pd.DataFrame,
        feature_columns: list[str],
        future_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        return self._dfs.build_future_rows_autoregressive(
            hist_df, feature_columns, future_dates
        )

    # ...
    # (Altre funzioni di logging/copertura mensile omesse nella versione ridotta)
    # ...
