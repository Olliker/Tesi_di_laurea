from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecasting.app.models import PredictionResult, ShapExplanation


class DataNormalizeService:
    def translate_multiple_predictions_to_text(
        self, xgboost_response: PredictionResult
    ) -> List[str]:
        fp = xgboost_response.future_predictions
        print(
            f"[normalize] future_predictions type: {type(fp)}, keys: {fp.keys() if hasattr(fp, 'keys') else 'N/A'}",
            flush=True,
        )
        if not fp or "article_id" not in fp:
            print(
                "[normalize] WARNING: No article_id in future_predictions, returning empty",
                flush=True,
            )
            return []

        df_raw, _ = self._build_prediction_dataframe(fp)
        if df_raw is None or df_raw.empty:
            return []

        df = df_raw.sort_values(["article_id", "prediction_date"]).reset_index(drop=True)

        texts: List[str] = []
        for aid, group in df.groupby("article_id", sort=False):
            if group.empty:
                continue

            group = group.sort_values("prediction_date").reset_index(drop=True)

            start_date = group["prediction_date"].iloc[0]
            end_date = group["prediction_date"].iloc[-1]

            article_display = self._format_article_display(aid, group)

            total_qty = float(group["predicted_quantity"].sum())
            avg_qty = float(group["predicted_quantity"].mean())
            min_qty = float(group["predicted_quantity"].min())
            max_qty = float(group["predicted_quantity"].max())
            first_qty = float(group["predicted_quantity"].iloc[0])
            last_qty = float(group["predicted_quantity"].iloc[-1])

            peak_idx = group["predicted_quantity"].idxmax()
            peak_qty = float(group.loc[peak_idx, "predicted_quantity"])
            peak_date = group.loc[peak_idx, "prediction_date"]

            range_text = self._format_period_range(start_date, end_date)
            level_desc = self._describe_demand_level(avg_qty)
            trend_desc = self._describe_trend(first_qty, last_qty)
            horizon_text = self._describe_horizon(start_date, end_date, len(group))
            periods_total = len(group)

            for idx_row, row in group.iterrows():
                pred_date = row["prediction_date"]
                qty = float(row["predicted_quantity"])
                texts.append(
                    (
                        f"Per l'articolo {article_display}, il {pred_date.strftime('%d %B %Y')} "
                        f"(finestra {idx_row + 1}/{periods_total} nel periodo {range_text}) prevediamo {qty:.1f} unità. "
                        f"Sull'intero intervallo stimiamo {total_qty:.1f} unità totali con una media di {avg_qty:.1f} ({level_desc}), "
                        f"valori compresi tra {min_qty:.1f} e {max_qty:.1f}. Il picco raggiunge {peak_qty:.1f} unità il "
                        f"{peak_date.strftime('%d %B %Y')} e la dinamica della domanda risulta {trend_desc} su {horizon_text}."
                    )
                )

        return texts

    def translate_multiple_explanations_to_text(
        self,
        shap_explanation: ShapExplanation,
        feature_names: List[str],
        xgboost_response: PredictionResult,
    ) -> List[str]:
        """Traduce le spiegazioni SHAP in testo normalizzato per ogni predizione"""
        if not hasattr(shap_explanation, "values") or shap_explanation.values is None:
            return ["Nessuna spiegazione SHAP disponibile"]
        if shap_explanation.values.size == 0:
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
        pos_to_shap_idx = {
            int(pos): idx for idx, pos in enumerate(row_positions.tolist())
        }
        df["__shap_idx"] = df["__row_pos"].map(pos_to_shap_idx)

        translated_texts = []
        for aid, group in df.groupby("article_id", sort=False):
            if group.empty:
                continue

            group = group.sort_values("prediction_date").reset_index(drop=True)
            article_display = self._format_article_display(aid, group)

            for _, row in group.iterrows():
                shap_idx_val = row.get("__shap_idx")
                agg_shap = np.asarray([])
                if pd.notna(shap_idx_val):
                    shap_pos = int(shap_idx_val)
                    if 0 <= shap_pos < shap_rows_raw.shape[0]:
                        agg_shap = shap_rows_raw[shap_pos]

                summary = self._summarize_feature_contributions(
                    np.asarray(agg_shap, dtype=float), feature_names
                )
                pred_date = row["prediction_date"]
                month_label = pred_date.strftime("%B %Y")
                translated_texts.append(
                    f"Spiegazione per l'articolo {article_display} per il mese di {month_label}: {summary}"
                )

        return translated_texts

    def translate_shap_to_text(
        self,
        shap_response: ShapExplanation,
        feature_names: Optional[List[str]] = None,
        *,
        row_index: Optional[int] = None,
        top_n: int = 3,
    ) -> str:
        """Traduce le spiegazioni SHAP in linguaggio testuale"""
        if not hasattr(shap_response, "values") or shap_response.values is None:
            return "Nessuna spiegazione SHAP disponibile"

        if shap_response.values.size == 0:
            return "Nessuna spiegazione SHAP disponibile"

        shap_values = shap_response.values
        try:
            if shap_values.ndim > 1:
                if row_index is not None and 0 <= row_index < shap_values.shape[0]:
                    shap_values = shap_values[row_index]
                else:
                    shap_values = shap_values.mean(axis=0)
        except Exception:
            shap_values = np.asarray(shap_values).flatten()

        shap_values = np.asarray(shap_values, dtype=float).flatten()

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(shap_values))]
        summary = self._summarize_feature_contributions(
            shap_values, feature_names, top_n=top_n
        )
        return summary

    def build_embedding_documents(
        self,
        prediction_result: PredictionResult,
        shap_explanation: ShapExplanation,
        feature_names: List[str],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        fp = prediction_result.future_predictions
        df_raw, row_positions = self._build_prediction_dataframe(fp)
        if df_raw is None or df_raw.empty:
            return [], []

        shap_rows_raw = self._select_shap_rows(shap_explanation.values, row_positions)
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        df = df_raw.sort_values(["article_id", "prediction_date"]).reset_index(drop=True)
        pos_to_shap_idx = {
            int(pos): idx for idx, pos in enumerate(row_positions.tolist())
        }
        df["__shap_idx"] = df["__row_pos"].map(pos_to_shap_idx)

        documents = []
        metadatas = []

        for aid, group in df.groupby("article_id", sort=False):
            if group.empty:
                continue

            group = group.sort_values("prediction_date").reset_index(drop=True)
            article_display = self._format_article_display(aid, group)
            start_date = group["prediction_date"].min()
            end_date = group["prediction_date"].max()
            range_text = self._format_period_range(start_date, end_date)
            total_qty = float(group["predicted_quantity"].sum())
            avg_qty = float(group["predicted_quantity"].mean())
            min_qty = float(group["predicted_quantity"].min())
            max_qty = float(group["predicted_quantity"].max())
            first_qty = float(group["predicted_quantity"].iloc[0])
            last_qty = float(group["predicted_quantity"].iloc[-1])
            trend_desc = self._describe_trend(first_qty, last_qty)
            periods_total = len(group)

            for idx_row, row in group.iterrows():
                shap_idx_val = row.get("__shap_idx")
                agg_shap = np.asarray([])
                if pd.notna(shap_idx_val):
                    shap_pos = int(shap_idx_val)
                    if 0 <= shap_pos < shap_rows_raw.shape[0]:
                        agg_shap = shap_rows_raw[shap_pos]

                contributions = self._compute_feature_contributions(
                    np.asarray(agg_shap, dtype=float), feature_names
                )

                qty = float(row["predicted_quantity"])
                pred_date = row["prediction_date"]

                doc = (
                    f"Articolo: {article_display}\n"
                    f"Data previsione: {pred_date.strftime('%d %B %Y')}\n"
                    f"Quantità prevista: {qty:.1f}\n"
                    f"Posizione finestra: {idx_row + 1} di {periods_total}\n"
                    f"Periodo complessivo: {range_text}\n"
                    f"Metriche complessive articolo:\n"
                    f"  - Quantità totale prevista: {total_qty:.1f}\n"
                    f"  - Quantità media prevista: {avg_qty:.1f}\n"
                    f"  - Quantità minima prevista: {min_qty:.1f}\n"
                    f"  - Quantità massima prevista: {max_qty:.1f}\n"
                    f"  - Trend stimato sull'intero periodo: {trend_desc}\n"
                    f"Principali driver SHAP:\n"
                    + features_block
                )

                documents.append(doc)
                metadatas.append(
                    {
                        "type": "context_period",
                        "article_id": str(aid),
                        "article_name": self._extract_article_name(group),
                        "prediction_date": pred_date.strftime("%Y-%m-%d"),
                        "forecast_window_index": idx_row + 1,
                        "forecast_windows_total": periods_total,
                        "predicted_quantity": qty,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "total_quantity": total_qty,
                        "average_quantity": avg_qty,
                        "min_quantity": min_qty,
                        "max_quantity": max_qty,
                        "trend": trend_desc,
                        "top_features": "; ".join(
                            f"{name} ({'+' if val > 0 else '-'}{abs(val):.3f}) {direction}"
                            for name, val, direction in contributions
                        ),
                    }
                )

        return documents, metadatas

    @staticmethod
    def _format_period_range(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
        if pd.isna(start_date) and pd.isna(end_date):
            return "nel periodo di previsione"
        if pd.isna(start_date):
            return f"fino al {end_date.strftime('%d %B %Y')}"
        if pd.isna(end_date):
            return f"a partire dal {start_date.strftime('%d %B %Y')}"
        if start_date.date() == end_date.date():
            return f"per il {start_date.strftime('%d %B %Y')}"
        if start_date.month == end_date.month and start_date.year == end_date.year:
            return (
                f"nel periodo dal {start_date.strftime('%d')} al {end_date.strftime('%d %B %Y')}"
            )
        return (
            f"nel periodo dal {start_date.strftime('%d %B %Y')} al {end_date.strftime('%d %B %Y')}"
        )

    @staticmethod
    def _describe_demand_level(avg_qty: float) -> str:
        if avg_qty >= 100:
            return "domanda molto elevata"
        if avg_qty >= 50:
            return "domanda sostenuta"
        if avg_qty >= 15:
            return "domanda moderata"
        if avg_qty > 1:
            return "domanda contenuta"
        return "domanda trascurabile"

    @staticmethod
    def _describe_trend(first_qty: float, last_qty: float) -> str:
        base = max(abs(first_qty), 1.0)
        change = last_qty - first_qty
        ratio = change / base
        if ratio > 0.35:
            return "in crescita"
        if ratio > 0.1:
            return "in lieve crescita"
        if ratio < -0.35:
            return "in forte calo"
        if ratio < -0.1:
            return "in diminuzione"
        return "quasi stabile"

    @staticmethod
    def _describe_horizon(
        start_date: pd.Timestamp, end_date: pd.Timestamp, periods: int
    ) -> str:
        if pd.isna(start_date) or pd.isna(end_date):
            return "un orizzonte di previsione non definito"

        if end_date < start_date:
            start_date, end_date = end_date, start_date

        horizon_days = int((end_date - start_date).days)
        # includi il giorno iniziale nel conteggio se positivo
        if horizon_days >= 0:
            horizon_days += 1
        horizon_days = max(horizon_days, 1)

        if periods <= 1:
            intervals_text = "una singola finestra mensile"
        elif periods == 2:
            intervals_text = "due finestre mensili consecutive"
        else:
            intervals_text = f"{periods} finestre mensili consecutive"

        month_span = (
            (end_date.year - start_date.year) * 12
            + (end_date.month - start_date.month)
            + 1
        )
        month_span = max(month_span, 1)

        if month_span == 1:
            horizon_text = "1 mese"
        else:
            horizon_text = f"{month_span} mesi"

        return f"{intervals_text} su circa {horizon_text}"

    @staticmethod
    def _summarize_feature_contributions(
        shap_values: np.ndarray,
        feature_names: List[str],
        *,
        top_n: int = 3,
        threshold: float = 0.01,
    ) -> str:
        contributions = DataNormalizeService._compute_feature_contributions(
            shap_values, feature_names, threshold=threshold
        )

        if not contributions:
            return "Nessuna feature ha influenzato significativamente la predizione"

        top = contributions[: max(1, top_n)]

        parts = [
            f"'{name}' ha {direction} la previsione di {abs(value):.3f}"
            for name, value, direction in top
        ]
        return "Le feature con impatto maggiore sono: " + "; ".join(parts)

    @staticmethod
    def _compute_feature_contributions(
        shap_values: np.ndarray,
        feature_names: List[str],
        *,
        threshold: float = 0.01,
    ) -> List[Tuple[str, float, str]]:
        contributions: List[Tuple[str, float, str]] = []
        for name, importance in zip(feature_names, shap_values):
            if abs(importance) >= threshold:
                direction = "aumentato" if importance > 0 else "ridotto"
                contributions.append((name, float(importance), direction))

        contributions.sort(key=lambda item: abs(item[1]), reverse=True)
        return contributions

    @staticmethod
    def _build_prediction_dataframe(
        future_predictions: Dict[str, Any]
    ) -> Tuple[Optional[pd.DataFrame], np.ndarray]:
        try:
            df = pd.DataFrame.from_dict(future_predictions)
        except Exception as exc:
            print(
                f"[normalize] WARNING: unable to build dataframe from predictions: {exc}",
                flush=True,
            )
            return None, np.array([], dtype=int)

        required_cols = {"article_id", "prediction_date"}
        if df.empty or not required_cols.issubset(df.columns):
            return None, np.array([], dtype=int)

        df = df.reset_index(drop=True)
        df["__row_pos"] = df.index.astype(int)

        df["prediction_date"] = pd.to_datetime(df["prediction_date"], errors="coerce")
        if "predicted_quantity" in df.columns:
            df["predicted_quantity"] = pd.to_numeric(
                df["predicted_quantity"], errors="coerce"
            ).fillna(0.0)
        else:
            df["predicted_quantity"] = 0.0

        valid_mask = df["article_id"].notna() & df["prediction_date"].notna()
        df = df.loc[valid_mask].copy()
        if df.empty:
            return None, np.array([], dtype=int)

        row_positions = df["__row_pos"].to_numpy(dtype=int)
        df.reset_index(drop=True, inplace=True)
        return df, row_positions

    @staticmethod
    def _select_shap_rows(shap_values: Any, row_positions: np.ndarray) -> np.ndarray:
        if shap_values is None or row_positions.size == 0:
            return np.asarray([])

        shap_matrix = np.asarray(shap_values, dtype=float)
        shap_matrix = np.atleast_2d(shap_matrix)
        total_rows = shap_matrix.shape[0]
        if total_rows == 0:
            return np.asarray([])

        if total_rows == 1 and row_positions.size > 1:
            return np.repeat(shap_matrix, row_positions.size, axis=0)

        indices: List[int] = []
        last_idx = max(total_rows - 1, 0)
        for pos in row_positions:
            pos_int = int(pos)
            idx = pos_int if 0 <= pos_int < total_rows else last_idx
            indices.append(idx)

        if not indices:
            return np.asarray([])

        return shap_matrix[np.asarray(indices, dtype=int)]

    @staticmethod
    def _format_article_display(aid: Any, group: pd.DataFrame) -> str:
        aid_str = str(aid)
        article_name = DataNormalizeService._extract_article_name(group)
        if article_name:
            return f"{aid_str} - {article_name}"
        return aid_str

    @staticmethod
    def _extract_article_name(group: pd.DataFrame) -> str:
        if "article_name" not in group.columns:
            return ""
        names = group["article_name"].dropna()
        if names.empty:
            return ""
        return str(names.iloc[0])
