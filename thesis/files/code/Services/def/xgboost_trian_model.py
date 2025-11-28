def __prepare_training_data(
        self, cutoff_date: pd.Timestamp | None
    ) -> tuple[pd.DataFrame, list[str], str, str]:
        cutoff = (
            self.__as_naive_timestamp(cutoff_date)
            if cutoff_date is not None
            else None
        )
        if hasattr(self.__data_service, "prepare_regression_frame"):
            prepared, feature_cols, date_col, target_col = (
                self.__data_service.prepare_regression_frame(cutoff_date=cutoff)
            )
        else:
            prepared, feature_cols, date_col, target_col = (
                self.__data_service.prepare_dataset(cutoff_date=cutoff)
            )
        if prepared.empty:
            raise ValueError("Nessun dato disponibile per l'addestramento del modello.")
        prepared = prepared.sort_values(date_col)
        return prepared, feature_cols, date_col, 

def train_model(self, cutoff_date: pd.Timestamp | None = None) -> None:
        prepared, feature_cols, date_col, target_col = self.__prepare_training_data(
            cutoff_date
        )
        self.__fit_model(
            prepared=prepared,
            feature_cols=feature_cols,
            date_col=date_col,
            target_col=target_col,
        )