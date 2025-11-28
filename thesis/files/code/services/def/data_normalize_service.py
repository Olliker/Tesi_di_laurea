class DataNormalizeService:
    """
    Versione ridotta per la tesi: traduce previsioni e spiegazioni in testo
    e costruisce documenti per l'embedding.
    """

    def translate_multiple_predictions_to_text(
        self, xgboost_response: PredictionResult
    ) -> list[str]:
        fp = xgboost_response.future_predictions
        df_raw, _ = self._build_prediction_dataframe(fp)
        if df_raw is None or df_raw.empty:
            return []
        df = df_raw.sort_values(["article_id", "prediction_date"]).reset_index(drop=True)

        texts: list[str] = []
        for aid, group in df.groupby("article_id", sort=False):
            if group.empty:
                continue
            group = group.sort_values("prediction_date").reset_index(drop=True)
            article_display = self._format_article_display(aid, group)
            for idx_row, row in group.iterrows():
                pred_date = row["prediction_date"]
                qty = float(row["predicted_quantity"])
                texts.append(
                    f"Per l'articolo {article_display}, il {pred_date:%d %B %Y} "
                    f"(finestra {idx_row + 1}/{len(group)}) prevediamo {qty:.1f} unità."
                )
        return texts

    def translate_multiple_explanations_to_text(
        self,
        shap_explanation: ShapExplanation,
        feature_names: list[str],
        xgboost_response: PredictionResult,
    ) -> list[str]:
        if not hasattr(shap_explanation, "values") or shap_explanation.values is None:
            return ["Nessuna spiegazione SHAP disponibile"]
        df_raw, row_positions = self._build_prediction_dataframe(
            xgboost_response.future_predictions
        )
        if df_raw is None or df_raw.empty or row_positions.size == 0:
            return ["Nessuna spiegazione SHAP disponibile"]

        shap_rows_raw = self._select_shap_rows(shap_explanation.values, row_positions)
        if shap_rows_raw.size == 0:
            return ["Nessuna spiegazione SHAP disponibile"]

        df = df_raw.sort_values(["article_id", "prediction_date"]).reset_index(drop=True)
        pos_to_shap_idx = {int(pos): idx for idx, pos in enumerate(row_positions.tolist())}
        df["__shap_idx"] = df["__row_pos"].map(pos_to_shap_idx)

        translated: list[str] = []
        for aid, group in df.groupby("article_id", sort=False):
            if group.empty:
                continue
            article_display = self._format_article_display(aid, group)
            for _, row in group.iterrows():
                shap_idx_val = row.get("__shap_idx")
                shap_vec = np.asarray([])
                if pd.notna(shap_idx_val):
                    shap_pos = int(shap_idx_val)
                    if 0 <= shap_pos < shap_rows_raw.shape[0]:
                        shap_vec = shap_rows_raw[shap_pos]
                summary = self._summarize_feature_contributions(
                    np.asarray(shap_vec, dtype=float), feature_names
                )
                pred_date = row["prediction_date"]
                translated.append(
                    f"Spiegazione per l'articolo {article_display} ({pred_date:%B %Y}): {summary}"
                )
        return translated

    def build_embedding_documents(
        self,
        prediction_result: PredictionResult,
        shap_explanation: ShapExplanation,
        feature_names: list[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        fp = prediction_result.future_predictions
        df_raw, row_positions = self._build_prediction_dataframe(fp)
        if df_raw is None or df_raw.empty:
            return [], []

        shap_rows_raw = self._select_shap_rows(shap_explanation.values, row_positions)
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        df = df_raw.sort_values(["article_id", "prediction_date"]).reset_index(drop=True)
        pos_to_shap_idx = {int(pos): idx for idx, pos in enumerate(row_positions.tolist())}
        df["__shap_idx"] = df["__row_pos"].map(pos_to_shap_idx)

        for aid, group in df.groupby("article_id", sort=False):
            if group.empty:
                continue
            group = group.sort_values("prediction_date").reset_index(drop=True)
            article_display = self._format_article_display(aid, group)
            for idx_row, row in group.iterrows():
                shap_idx_val = row.get("__shap_idx")
                shap_vec = np.asarray([])
                if pd.notna(shap_idx_val):
                    shap_pos = int(shap_idx_val)
                    if 0 <= shap_pos < shap_rows_raw.shape[0]:
                        shap_vec = shap_rows_raw[shap_pos]
                contributions = self._compute_feature_contributions(
                    np.asarray(shap_vec, dtype=float), feature_names
                )
                qty = float(row["predicted_quantity"])
                pred_date = row["prediction_date"]
                doc = (
                    f"Articolo: {article_display}\\n"
                    f"Data previsione: {pred_date:%d %B %Y}\\n"
                    f"Quantità prevista: {qty:.1f}\\n"
                    f"Principali driver SHAP:\\n"
                    + "\\n".join(
                        f\"- {name}: {'+' if val > 0 else '-'}{abs(val):.3f} ({direction})\"
                        for name, val, direction in contributions
                    )
                )
                documents.append(doc)
                metadatas.append(
                    {
                        "type": "context_period",
                        "article_id": str(aid),
                        "article_name": self._extract_article_name(group),
                        "prediction_date": pred_date.strftime("%Y-%m-%d"),
                        "forecast_window_index": idx_row + 1,
                        "forecast_windows_total": len(group),
                        "predicted_quantity": qty,
                    }
                )

        return documents, metadatas

    # (Restanti helper: _build_prediction_dataframe, _select_shap_rows,
    # _format_article_display, _extract_article_name, _summarize_feature_contributions, etc.)
