def __prepare_training_data(
    self, cutoff_date: pd.Timestamp | None
) -> tuple[pd.DataFrame, list[str], str, str]:
    """
    Prepara il dataset di regressione: carica dati, applica cutoff temporale,
    restituisce frame ordinato, feature, colonna data e target.
    """
    prepared, feature_cols, date_col, target_col = (
        self.__data_service.prepare_regression_frame(cutoff_date=cutoff_date)
    )
    if prepared.empty:
        raise ValueError("Nessun dato disponibile per l'addestramento.")
    return prepared.sort_values(date_col), feature_cols, date_col, target_col


def train_model(self, cutoff_date: pd.Timestamp | None = None) -> None:
    """
    Flusso di training semplificato:
    1. prepara dati e feature;
    2. imputazione + split temporale 80/20;
    3. fit del modello XGBoost;
    4. calcolo WAPE% su hold-out (se presente).
    """
    prepared, feature_cols, date_col, target_col = self.__prepare_training_data(
        cutoff_date
    )
    self.__fit_model(
        prepared=prepared,
        feature_cols=feature_cols,
        date_col=date_col,
        target_col=target_col,
    )
