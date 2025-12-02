

class XGBoostService:

    def __init__(
        self,
        *,
        metrics: MetricsService | None = None,
        data_service: ForecastingDataService | None = None,
        config: XGBoostConfig | None = None,
    ):
        self.__cfg = config or XGBoostConfig()
        self.__model = xgb.XGBRegressor(**self.__params_from_config(self.__cfg))
        self.__data_service = data_service
        self.__metrics = metrics or MetricsService()
        self.feature_columns: List[str] | None = None
        self.__last_wape: float | None = None

    # --------------------------- util ---------------------------

    def __as_naive_timestamp(self, dt: datetime | pd.Timestamp) -> pd.Timestamp:
        ts = pd.to_datetime(dt)
        return ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts

    def __params_from_config(self, cfg: XGBoostConfig) -> dict:
        # CHANGED: evitiamo 'mape' come eval_metric sul log; usiamo rmse
        params = {
            "n_estimators": cfg.n_estimators,
            "learning_rate": cfg.learning_rate,
            "max_depth": cfg.max_depth,
            "n_jobs": cfg.n_jobs,
            "min_child_weight": cfg.min_child_weight,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_lambda": cfg.reg_lambda,
            "reg_alpha": cfg.reg_alpha,
            "random_state": cfg.random_state,
            "tree_method": cfg.tree_method,
            "max_bin": cfg.max_bin,
            "objective": cfg.objective,
            "eval_metric": "rmse",  # <--- CHANGED
            "verbosity": getattr(cfg, "verbosity", 1),
        }
        return params

    def __inverse_target(self, yhat: np.ndarray) -> np.ndarray:
        transform = getattr(self.__cfg, "target_transform", TARGET_TRANSFORM)
        if transform == "log1p":
            return np.expm1(yhat)
        return yhat

    # --------------------------- training helpers ---------------------------

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
        return prepared, feature_cols, date_col, target_col

    def __impute_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """CHANGED: imputazione più robusta per evitare colonne tutte zero nel futuro."""
        X = X.copy()
        # numeriche -> mediana; non numeriche (se presenti) -> 0
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                med = X[col].median()
                if pd.isna(med):
                    med = 0.0
                X[col] = X[col].fillna(med)
            else:
                X[col] = X[col].fillna(0)
        return X

    def __fit_model(
        self,
        *,
        prepared: pd.DataFrame,
        feature_cols: list[str],
        date_col: str,
        target_col: str,
    ) -> None:
        # CHANGED: imputazione migliore
        X = self.__impute_features(prepared[feature_cols])
        y = prepared[target_col].astype(float).clip(lower=0.0)
        if X.empty:
            raise ValueError("Dataset senza feature utili per l'addestramento.")

        model = xgb.XGBRegressor(**self.__params_from_config(self.__cfg))

        y_full = y.values.astype(float)
        transform = getattr(self.__cfg, "target_transform", TARGET_TRANSFORM)

        if len(X) < 3:
            model.fit(X, y_full)
            self.__model = model
            self.feature_columns = feature_cols
            self.__last_wape = None
            print(
                "[xgb] train rows:",
                len(X),
                "val rows: 0 | training senza holdout (dati insufficienti)",
                flush=True,
            )
            return

        split_idx = max(1, int(len(X) * 0.8))
        if split_idx >= len(X):
            split_idx = len(X) - 1

        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        if transform == "log1p":
            y_train_t = np.log1p(y_train.values)
            y_val_t = np.log1p(y_val.values)
        else:
            y_train_t = y_train.values.astype(float)
            y_val_t = y_val.values.astype(float)

        fit_kwargs: dict[str, object] = dict(X=X_train, y=y_train_t)
        if not X_val.empty:
            fit_kwargs["eval_set"] = [(X_val, y_val_t)]
            fit_kwargs["verbose"] = False

        model.fit(**fit_kwargs)

        self.__model = model
        self.feature_columns = feature_cols

        if not X_val.empty:
            y_val_pred_raw = model.predict(X_val).astype(float)
            y_val_pred = np.maximum(0.0, self.__inverse_target(y_val_pred_raw))
            wape_val = self.__metrics.wape_percent(y_val.values, y_val_pred)
            self.__last_wape = self.__metrics.clean(wape_val)
            print(
                "[xgb] train rows:",
                len(X_train),
                "val rows:",
                len(X_val),
                "| target stats val:",
                y_val.describe().to_dict(),
                flush=True,
            )
            print(
                f"[xgb] Validation WAPE%: {self.__last_wape:.2f}",
                flush=True,
            )
        else:
            self.__last_wape = None
            print(
                "[xgb] train rows:",
                len(X_train),
                "val rows: 0 | holdout non disponibile",
                flush=True,
            )

    def __build_monthly_dates(
        self, anchor: pd.Timestamp, count: int
    ) -> pd.DatetimeIndex:
        if count <= 0:
            return pd.DatetimeIndex([])
        start = anchor.to_period("M").to_timestamp()
        return pd.date_range(start=start, periods=count, freq="MS")

    def __build_real_predictions(
        self,
        *,
        prepared: pd.DataFrame,
        start_ts: pd.Timestamp,
        periods: int,
        use_actual_if_available: bool = True,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if not self.feature_columns:
            raise RuntimeError("feature_columns non inizializzate")

        date_col = self.__data_frame_service.get_date_col()
        tgt = self.__data_frame_service.get_target_col()
        price_cols = self.__data_service.get_price_columns(include_derivatives=True)
        feature_cols = list(self.feature_columns)

        future_dates = self.__build_monthly_dates(start_ts, periods)
        if future_dates.empty:
            return (
                pd.DataFrame(
                    columns=[
                        "article_id",
                        "article_name",
                        "prediction_date",
                        "predicted_quantity",
                    ]
                ),
                np.empty((0, len(feature_cols))),
            )

        df = prepared.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        records: list[dict[str, object]] = []
        feature_vectors: list[np.ndarray] = []

        grouped = df.sort_values(date_col).groupby("article_id", sort=False)
        for aid, hist in grouped:
            if hist.empty:
                continue
            hist_sorted = hist.sort_values(date_col).copy()
            hist_sorted[date_col] = pd.to_datetime(hist_sorted[date_col], errors="coerce")
            hist_sorted = (
                hist_sorted.dropna(subset=[date_col])
                .sort_values(date_col)
                .reset_index(drop=True)
            )
            if hist_sorted.empty:
                continue

            article_name = (
                hist_sorted["article_name"].dropna().iloc[-1]
                if "article_name" in hist_sorted.columns
                and not hist_sorted["article_name"].dropna().empty
                else f"Articolo_{aid}"
            )

            hist_dates = set(hist_sorted[date_col].to_list())

            for dt in future_dates:
                # CHANGED: non usiamo più i reali come "previsione" a meno di esplicita richiesta
                if dt in hist_dates and use_actual_if_available:
                    source_row = hist_sorted.loc[hist_sorted[date_col] == dt].iloc[-1]
                    use_actual = True
                else:
                    future_slice = self.__data_service.build_future_rows(
                        hist_sorted,
                        feature_cols,
                        pd.DatetimeIndex([dt]),
                    )
                    if future_slice.empty:
                        continue
                    row = future_slice.iloc[-1].copy()
                    row[date_col] = dt
                    row["article_id"] = aid
                    if "article_name" not in row or pd.isna(row["article_name"]):
                        row["article_name"] = article_name
                    source_row = row
                    use_actual = False

                # features
                features_series = source_row.reindex(feature_cols, fill_value=np.nan)
                features_df = pd.DataFrame([features_series.values], columns=feature_cols)
                features_df = self.__impute_features(features_df)
                features_numeric = features_df.iloc[0].to_numpy(dtype=float)
                if features_numeric.size != len(feature_cols):
                    continue

                if use_actual:
                    predicted_qty = float(source_row.get(tgt, 0.0))
                else:
                    y_pred_raw = self.__model.predict(
                        features_numeric.reshape(1, -1)
                    ).astype(float)[0]
                    predicted_qty = float(
                        np.maximum(
                            0.0, self.__inverse_target(np.array([y_pred_raw]))[0]
                        )
                    )

                record: dict[str, object] = {
                    "article_id": int(aid),
                    "article_name": str(
                        source_row.get("article_name", article_name)
                    ),
                    "prediction_date": dt,
                    "predicted_quantity": predicted_qty,
                }
                if "article_monthly_mean" in source_row.index:
                    try:
                        record["article_monthly_mean"] = float(
                            source_row.get("article_monthly_mean", 0.0)
                        )
                    except Exception:
                        record["article_monthly_mean"] = 0.0
                for col in price_cols:
                    if col in source_row.index:
                        try:
                            record[col] = float(source_row.get(col, 0.0))
                        except (TypeError, ValueError):
                            record[col] = 0.0
                records.append(record)

                feature_vectors.append(features_numeric)

                if not use_actual:
                    # autoregressione: aggiungiamo la riga prevista per creare lag/rolling credibili
                    new_row = source_row.copy()
                    new_row[tgt] = predicted_qty
                    new_row["article_id"] = aid
                    if "article_name" not in new_row or pd.isna(new_row["article_name"]):
                        new_row["article_name"] = article_name
                    if "sold_this_month" in new_row.index:
                        try:
                            new_row["sold_this_month"] = int(predicted_qty > 0.0)
                        except Exception:
                            new_row["sold_this_month"] = int(predicted_qty > 0.0)
                    month_avg_val = float(new_row.get("article_monthly_mean", 0.0))
                    if "article_monthly_mean" not in new_row.index:
                        month_avg_val = (
                            float(hist_sorted[tgt].astype(float).mean())
                            if not hist_sorted.empty
                            else 0.0
                        )
                        new_row["article_monthly_mean"] = month_avg_val
                    if month_avg_val > 1e-8:
                        seasonal_ratio = predicted_qty / month_avg_val
                    else:
                        seasonal_ratio = float(new_row.get("seasonal_ratio", 1.0))
                    new_row["seasonal_ratio"] = float(
                        0.0 if not np.isfinite(seasonal_ratio) else seasonal_ratio
                    )
                    new_row["summer_shutdown"] = int(dt.month == 8)
                    new_row["post_shutdown_recovery"] = int(dt.month == 9)
                    hist_sorted = pd.concat(
                        [hist_sorted, pd.DataFrame([new_row])], ignore_index=True
                    )
                    hist_sorted[date_col] = pd.to_datetime(
                        hist_sorted[date_col], errors="coerce"
                    )
                    hist_sorted = (
                        hist_sorted.dropna(subset=[date_col])
                        .sort_values(date_col)
                        .reset_index(drop=True)
                    )

        if not records or not feature_vectors:
            return (
                pd.DataFrame(
                    columns=[
                        "article_id",
                        "article_name",
                        "prediction_date",
                        "predicted_quantity",
                    ]
                ),
                np.empty((0, len(feature_cols))),
            )

        X_pred = np.vstack(feature_vectors)

        result = pd.DataFrame(records)
        result = result.sort_values(["article_id", "prediction_date"]).reset_index(
            drop=True
        )
        return result, X_pred

    # --------------------------- API pubbliche ---------------------------

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

    def get_model(self) -> xgb.XGBRegressor:
        return self.__model

    def predict(
        self,
        *,
        start_date: datetime,
        periods: int,
        use_actual_if_available: bool = False,  # CHANGED: esposto anche qui
    ) -> Tuple[PredictionResult, np.ndarray, List[str]]:
        if periods <= 0:
            raise ValueError("periods must be greater than zero")

        start_ts = self.__as_naive_timestamp(start_date)
        prediction_start = start_ts.to_period("M").to_timestamp()
        self.train_and_test_model(cutoff_date=prediction_start)

        if hasattr(self.__data_service, "prepare_regression_frame"):
            prepared_all, _, _, _ = self.__data_service.prepare_regression_frame()
        else:
            prepared_all, _, _, _ = self.__data_service.prepare_dataset()

        preds_df, X_pred_all = self.__build_real_predictions(
            prepared=prepared_all,
            start_ts=prediction_start,
            periods=periods,
            use_actual_if_available=use_actual_if_available,  # default False
        )

        # diagnostica base per capire “piattezza” delle feature future
        try:
            if X_pred_all.size > 0 and self.feature_columns:
                stds = X_pred_all.std(axis=0)
                near_const = [
                    c for c, s in zip(self.feature_columns, stds) if float(s) < 1e-9
                ]
                if near_const:
                    print("[xgb] Feature quasi costanti nel futuro:", near_const[:20], flush=True)
        except Exception:
            pass

        mape_value = self.__metrics.clean(self.__last_wape) if self.__last_wape is not None else None
        return (
            PredictionResult(
                future_predictions=preds_df.to_dict(),
                MAPE=mape_value if mape_value is not None else 99.9,
            ),
            X_pred_all,
            list(self.feature_columns or []),
        )
