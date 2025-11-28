def predict(
    self,
    *,
    start_date: datetime,
    periods: int,
    use_actual_if_available: bool = False,
) -> tuple[PredictionResult, np.ndarray, list[str]]:
    """
    Flusso di previsione:
    1. addestra il modello fino al mese precedente a start_date;
    2. genera date future e feature autoregressive;
    3. produce quantità previste (o reali se richiesto);
    4. restituisce PredictionResult, matrice feature e lista colonne.
    """
    start_ts = self.__as_naive_timestamp(start_date).to_period("M").to_timestamp()
    self.train_and_test_model(cutoff_date=start_ts)

    prepared_all, _, _, _ = self.__data_service.prepare_regression_frame()
    preds_df, X_pred = self.__build_real_predictions(
        prepared=prepared_all,
        start_ts=start_ts,
        periods=periods,
        use_actual_if_available=use_actual_if_available,
    )

    mape_value = self.__metrics.clean(self.__last_wape) if self.__last_wape is not None else None
    return (
        PredictionResult(
            future_predictions=preds_df.to_dict(),
            MAPE=mape_value if mape_value is not None else 99.9,
        ),
        X_pred,
        list(self.feature_columns or []),
    )
