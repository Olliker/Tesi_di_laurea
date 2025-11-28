from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from forecasting.app.services.data_frame_service import DataFrameService
from forecasting.app.services.invoice_conversion_service import (
    InvoiceConversionService,
)


class ForecastingDataService:
    def __init__(
        self,
        *,
        data_frame_service: DataFrameService | None = None,
        invoice_service: InvoiceConversionService | None = None,
    ):
        self._dfs = data_frame_service or DataFrameService()
        self._invoice_service = invoice_service or InvoiceConversionService()

    # ---------- accessors ----------

    def get_data_frame_service(self) -> DataFrameService:
        return self._dfs

    def get_price_columns(self, include_derivatives: bool = False) -> list[str]:
        return self._dfs.get_price_columns(include_derivatives=include_derivatives)

    # ---------- dataset preparation ----------

    def load_fact_df(
        self, cutoff_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        df = self._invoice_service.convert_db_data_to_df()
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("[data] dataframe sorgente vuoto o non valido")

        date_col = self._dfs.get_date_col()
        if date_col not in df.columns:
            raise ValueError(f"[data] colonna data '{date_col}' assente nel dataframe")

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        if cutoff_date is not None:
            cutoff_ts = pd.Timestamp(cutoff_date)
            if cutoff_ts.tzinfo is not None:
                try:
                    cutoff_ts = cutoff_ts.tz_localize(None)
                except TypeError:
                    cutoff_ts = cutoff_ts.tz_convert(None)
            df = df[df[date_col] < cutoff_ts]
            if df.empty:
                raise ValueError(
                    "[data] nessun dato disponibile prima della data di inizio richiesta"
                )

        return df

    def _log_dataset_stats(
        self,
        *,
        label: str,
        frame: pd.DataFrame,
        date_col: str,
        target_col: str,
    ) -> None:
        if frame.empty:
            return
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        stats = (
            frame[target_col].describe().to_dict()
            if target_col in frame.columns
            else "n/a"
        )
        print(
            f"[data] {label} rows:",
            len(frame),
            "| range:",
            f"{dates.min()} → {dates.max()}",
            "| target stats:",
            stats,
            flush=True,
        )

    def _ensure_monthly_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        date_col = self._dfs.get_date_col()
        tgt = self._dfs.get_target_col()

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        frames: list[pd.DataFrame] = []
        for aid, grp in df.sort_values(date_col).groupby("article_id", sort=False):
            if grp.empty:
                continue
            start = grp[date_col].min().to_period("M")
            end = grp[date_col].max().to_period("M")
            monthly_index = pd.period_range(start, end, freq="M").to_timestamp()
            expanded = grp.set_index(date_col).reindex(monthly_index)
            expanded["article_id"] = aid
            if "article_name" in expanded.columns:
                name_series = grp["article_name"].dropna()
                fallback_name = (
                    name_series.iloc[-1] if not name_series.empty else f"Articolo_{aid}"
                )
                expanded["article_name"] = (
                    expanded["article_name"].ffill().bfill().fillna(fallback_name)
                )

            if tgt in expanded.columns:
                expanded[tgt] = expanded[tgt].fillna(0.0)

            numeric_cols = expanded.select_dtypes(include=["number"]).columns.tolist()
            for col in numeric_cols:
                if col == tgt:
                    continue
                expanded[col] = expanded[col].fillna(0.0)

            expanded = expanded.reset_index().rename(columns={"index": date_col})
            frames.append(expanded)

        if not frames:
            return df

        reindexed = pd.concat(frames, ignore_index=True)
        return reindexed.sort_values(["article_id", date_col])

    def prepare_regression_frame(
        self, cutoff_date: Optional[pd.Timestamp] = None
    ) -> Tuple[pd.DataFrame, list[str], str, str]:
        raw = self.load_fact_df(cutoff_date=cutoff_date)
        date_col = self._dfs.get_date_col()
        target_col = self._dfs.get_target_col()
        self._log_dataset_stats(
            label="raw",
            frame=raw,
            date_col=date_col,
            target_col=target_col,
        )

        prepared = self._dfs.prepare_features(raw)
        prepared = prepared.sort_values(date_col)
        self._log_dataset_stats(
            label="prepared",
            frame=prepared,
            date_col=date_col,
            target_col=target_col,
        )

        feature_cols = self._dfs.get_feature_columns(prepared)
        return prepared, feature_cols, date_col, target_col

    def prepare_classifier_frame(
        self, cutoff_date: Optional[pd.Timestamp] = None
    ) -> Tuple[pd.DataFrame, List[str], str, str]:
        raw = self.load_fact_df(cutoff_date=cutoff_date)
        covered = self._ensure_monthly_coverage(raw)
        date_col = self._dfs.get_date_col()
        target_col = self._dfs.get_target_col()
        self._log_dataset_stats(
            label="raw (classifier)",
            frame=covered,
            date_col=date_col,
            target_col=target_col,
        )

        prepared = self._dfs.prepare_features(covered)
        prepared = prepared.sort_values(date_col)
        prepared["will_sell"] = (
            prepared[target_col].fillna(0.0) > 0
        ).astype(int)
        self._log_dataset_stats(
            label="prepared (classifier)",
            frame=prepared,
            date_col=date_col,
            target_col="will_sell",
        )

        feature_cols = self._dfs.get_classifier_feature_columns(prepared)
        return prepared, feature_cols, date_col, "will_sell"

    def prepare_dataset(
        self, cutoff_date: Optional[pd.Timestamp] = None
    ) -> Tuple[pd.DataFrame, list[str], str, str]:
        return self.prepare_regression_frame(cutoff_date=cutoff_date)

    # ---------- future rows ----------

    def build_future_rows(
        self,
        hist_df: pd.DataFrame,
        feature_columns: list[str],
        future_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        return self._dfs.build_future_rows_autoregressive(
            hist_df, feature_columns, future_dates
        )
