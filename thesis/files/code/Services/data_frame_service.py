from typing import Tuple, List, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from forecasting.app.models.config.data_frame_config import DataFrameConfig


class DataFrameService:
    PRICE_COLUMNS = [
        "unit_price_discounted",
        "unit_price_list",
        "discount_percent",
        "total_before_discount",
        "total_after_discount",
        "vat_rate",
    ]

    def __init__(self):
        self.__config = (
            DataFrameConfig()
        )  # atteso: .date_col, .target_col, .exclude (lista opzionale)

    def get_date_col(self) -> str:
        return self.__config.date_col

    def get_target_col(self) -> str:
        return self.__config.target_col

    def get_price_columns(self, include_derivatives: bool = False) -> List[str]:
        cols = list(self.PRICE_COLUMNS)
        if include_derivatives:
            suffixes = ["_change_1", "_pct_change"]
            cols.extend(f"{col}{suffix}" for col in self.PRICE_COLUMNS for suffix in suffixes)
        return cols

    # ----------------- feature engineering (blocchi riutilizzabili) -----------------
    def __add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # colonne base
        if self.__config.date_col not in df.columns:
            raise ValueError(
                f"Data column '{self.__config.date_col}' not found in dataframe"
            )

        df[self.__config.date_col] = pd.to_datetime(
            df[self.__config.date_col], errors="coerce"
        )
        df["day_of_week"] = df[self.__config.date_col].dt.dayofweek
        df["month"] = df[self.__config.date_col].dt.month
        df["year"] = df[self.__config.date_col].dt.year
        df["quarter"] = df[self.__config.date_col].dt.quarter
        for date_col in ("expiring_date", "shipping_date", "creation_date"):
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        numeric_cols = {"quantity": 0.0, "customer_country_id": 0.0, "is_italy": 0.0}
        for col in self.PRICE_COLUMNS:
            numeric_cols[col] = 0.0
        for col, default in numeric_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

        return df

    def __add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["article_id", self.__config.date_col]).copy()
        if self.__config.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.__config.target_col}' not found for lags."
            )
        lag_list = list(self.__config.lag_windows)
        g = df.groupby("article_id", group_keys=False)
        for lag in lag_list:
            df[f"{self.__config.target_col}_lag_{lag}"] = g[
                self.__config.target_col
            ].shift(lag)
        # YoY diff/ratio rimossi per evitare feature quasi costanti
        return df

    def __add_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["article_id", self.__config.date_col]).copy()
        date_col = self.__config.date_col
        tgt = self.__config.target_col

        if "sold_this_month" not in df.columns:
            df["sold_this_month"] = (df[tgt].fillna(0.0) > 0).astype(int)
        else:
            df["sold_this_month"] = df["sold_this_month"].fillna(0).astype(int)

        def enrich(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            sold = group["sold_this_month"].astype(int).tolist()
            period_codes = (
                (group[date_col].dt.year * 12 + (group[date_col].dt.month - 1))
                .astype(int)
                .tolist()
            )
            if not period_codes:
                return group

            first_code = period_codes[0]
            months_since_first = [code - first_code for code in period_codes]

            months_with_sales = int(sum(sold))
            history_months = max(period_codes[-1] - first_code + 1, len(group))
            sales_ratio = months_with_sales / history_months if history_months > 0 else 0.0

            had_last = [0] + sold[:-1]
            had_year = [0] * len(group)
            for idx in range(12, len(group)):
                had_year[idx] = sold[idx - 12]

            months_since_last: list[int] = []
            last_sale_code: int | None = None
            for code, val in zip(period_codes, sold):
                if last_sale_code is None:
                    months_since_last.append(0 if val else code - first_code)
                else:
                    months_since_last.append(code - last_sale_code)
                if val:
                    last_sale_code = code

            group["months_since_first_sale"] = months_since_first
            group["months_since_last_sale"] = months_since_last
            group["months_with_sales"] = months_with_sales
            group["history_months"] = history_months
            group["sales_history_ratio"] = sales_ratio
            group["had_sales_last_month"] = had_last
            group["had_sales_last_year"] = had_year

            quantities = group[tgt].fillna(0.0).astype(float).tolist()
            running: list[float] = []
            cum_mean: list[float] = []
            cum_std: list[float] = []
            for value in quantities:
                running.append(float(value))
                arr = np.asarray(running, dtype=float)
                cum_mean.append(float(arr.mean()))
                cum_std.append(float(arr.std(ddof=0)) if arr.size > 1 else 0.0)
            group["quantity_cum_mean"] = cum_mean
            group["quantity_cum_std"] = cum_std
            return group

        groups = []
        for _, grp in df.groupby("article_id", sort=False):
            groups.append(enrich(grp))
        if groups:
            df = pd.concat(groups, ignore_index=True)
        else:
            df = df.iloc[0:0]
        return df

    def __add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["article_id", self.__config.date_col]).copy()
        tgt = self.__config.target_col
        g = df.groupby("article_id", group_keys=False)
        s = g[tgt].shift(1)
        roll_windows = list(self.__config.roll_windows)
        for w in roll_windows:
            df[f"{tgt}_roll_mean_{w}"] = s.rolling(
                window=w, min_periods=max(2, w // 2)
            ).mean()
            df[f"{tgt}_roll_std_{w}"] = s.rolling(
                window=w, min_periods=max(2, w // 2)
            ).std()
            df[f"{tgt}_roll_med_{w}"] = s.rolling(
                window=w, min_periods=max(2, w // 2)
            ).median()
        df["roc_1"] = (df[tgt] - g[tgt].shift(1)) / (np.abs(g[tgt].shift(1)) + 1e-7)
        df["roc_2"] = (g[tgt].shift(1) - g[tgt].shift(2)) / (
            np.abs(g[tgt].shift(2)) + 1e-7
        )
        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df[num_cols] = df[num_cols].fillna(0.0)
        return df

    def __add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["is_spring"] = ((df["month"] >= 3) & (df["month"] <= 5)).astype(int)
        df["is_summer"] = ((df["month"] >= 6) & (df["month"] <= 8)).astype(int)
        df["is_autumn"] = ((df["month"] >= 9) & (df["month"] <= 11)).astype(int)
        df["is_winter"] = ((df["month"] == 12) | (df["month"] <= 2)).astype(int)
        df["summer_shutdown"] = (df["month"] == 8).astype(int)
        df["post_shutdown_recovery"] = (df["month"] == 9).astype(int)
        return df

    def __add_monthly_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        date_col = self.__config.date_col
        tgt = self.__config.target_col
        if date_col not in df.columns or tgt not in df.columns:
            return df

        month_series = df[date_col].dt.month
        df["__month_tmp"] = month_series
        df["article_monthly_mean"] = (
            df.groupby(["article_id", "__month_tmp"])[tgt]
            .transform("mean")
            .fillna(0.0)
        )
        df["seasonal_ratio"] = df[tgt] / (df["article_monthly_mean"] + 1e-6)
        df["seasonal_ratio"] = (
            pd.to_numeric(df["seasonal_ratio"], errors="coerce")
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )
        df["article_avg_monthly_sales"] = (
            df.groupby("article_id")[tgt].transform("mean").fillna(0.0)
        )
        df["article_sales_variance"] = (
            df.groupby("article_id")[tgt].transform("std").fillna(0.0)
        )
        df.drop(columns="__month_tmp", inplace=True)
        return df

    def __add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        cols = [c for c in self.PRICE_COLUMNS if c in df.columns]
        if not cols:
            return df

        date_col = self.__config.date_col
        df = df.sort_values(["article_id", date_col]).copy()
        grouped = df.groupby("article_id", group_keys=False)

        for col in cols:
            filled = grouped[col].ffill().fillna(0.0)
            diff = grouped[col].diff().fillna(0.0)
            pct = (
                grouped[col]
                .pct_change()
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            df[col] = filled
            df[f"{col}_change_1"] = diff
            df[f"{col}_pct_change"] = pct

        return df

    def build_future_rows_autoregressive(
        self,
        hist_df: pd.DataFrame,
        feature_columns: List[str],
        future_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Costruisce righe future autoregressive aggiornando lag/rolling ad ogni passo
        usando anche le predizioni già calcolate.
        Restituisce un DataFrame con tutte le righe future (senza la colonna target).
        """

        date_col = self.__config.date_col
        target_col = self.__config.target_col

        lag_list = list(self.__config.lag_windows)
        roll_windows = list(self.__config.roll_windows)

        all_rows = []
        # ciclo per articolo
        for aid, g in hist_df.sort_values([date_col]).groupby("article_id"):
            hist = g.sort_values([date_col]).copy()
            article_name = (
                hist["article_name"].iloc[-1]
                if "article_name" in hist.columns and len(hist["article_name"]) > 0
                else f"Articolo_{aid}"
            )

            hist_month_series = hist[date_col].dt.month
            month_avg_map = (
                hist.assign(__month_tmp=hist_month_series)
                .groupby("__month_tmp")[target_col]
                .mean()
                .to_dict()
            )
            month_avg_default = (
                float(hist[target_col].astype(float).mean())
                if not hist.empty
                else 0.0
            )
            article_avg_sales = month_avg_default
            article_sales_var = (
                float(hist[target_col].astype(float).std(ddof=0))
                if len(hist) > 1
                else 0.0
            )
            if "seasonal_ratio" in hist.columns:
                ratio_series = hist["seasonal_ratio"].dropna()
                last_seasonal_ratio = (
                    float(ratio_series.iloc[-1]) if not ratio_series.empty else 1.0
                )
            else:
                last_seasonal_ratio = 1.0

            sold_series = (
                hist["sold_this_month"]
                if "sold_this_month" in hist.columns
                else (hist[target_col] > 0).astype(int)
            ).fillna(0).astype(int)
            period_codes = (
                hist[date_col].dt.year * 12 + (hist[date_col].dt.month - 1)
            ).astype(int)
            first_period_code = int(period_codes.iloc[0]) if len(period_codes) else None
            history_months_base = (
                int(period_codes.iloc[-1] - period_codes.iloc[0] + 1)
                if len(period_codes) > 0
                else 1
            )
            sale_period_codes = (
                set(period_codes[sold_series > 0].tolist())
                if len(period_codes)
                else set()
            )
            months_with_sales = int(sold_series.sum())
            last_sale_period = (
                max(sale_period_codes)
                if sale_period_codes
                else (int(period_codes.iloc[-1]) if len(period_codes) else None)
            )
            last_month_sales_flag = int(sold_series.iloc[-1]) if len(sold_series) else 0

            for dt in future_dates:
                # usa le stesse trasformazioni già fatte (temporal + seasonal)
                base = {
                    "article_id": aid,
                    "article_name": article_name,
                    date_col: dt,
                }
                hist_values = hist[target_col].astype(float).dropna()
                if not hist_values.empty:
                    base["quantity_cum_mean"] = float(hist_values.mean())
                    base["quantity_cum_std"] = float(hist_values.std(ddof=0))
                else:
                    base["quantity_cum_mean"] = 0.0
                    base["quantity_cum_std"] = 0.0
                period_dt_code = dt.year * 12 + (dt.month - 1)
                if first_period_code is not None:
                    base["months_since_first_sale"] = max(
                        0, period_dt_code - first_period_code
                    )
                    history_months_future = period_dt_code - first_period_code + 1
                else:
                    base["months_since_first_sale"] = 0
                    history_months_future = base["months_since_first_sale"] + 1

                if last_sale_period is not None:
                    base["months_since_last_sale"] = max(
                        0, period_dt_code - last_sale_period
                    )
                else:
                    base["months_since_last_sale"] = base["months_since_first_sale"]

                history_months_future = max(history_months_future, history_months_base)
                base["history_months"] = history_months_future
                base["months_with_sales"] = months_with_sales
                base["sales_history_ratio"] = (
                    months_with_sales / history_months_future
                    if history_months_future > 0
                    else 0.0
                )
                base["had_sales_last_month"] = last_month_sales_flag
                base["had_sales_last_year"] = (
                    1 if (period_dt_code - 12) in sale_period_codes else 0
                )
                base["sold_this_month"] = 0
                # temporal
                base["day_of_week"] = dt.dayofweek
                base["month"] = dt.month
                base["year"] = dt.year
                base["quarter"] = dt.quarter
                # seasonal
                base["month_sin"] = np.sin(2 * np.pi * base["month"] / 12)
                base["month_cos"] = np.cos(2 * np.pi * base["month"] / 12)
                base["day_of_week_sin"] = np.sin(2 * np.pi * base["day_of_week"] / 7)
                base["day_of_week_cos"] = np.cos(2 * np.pi * base["day_of_week"] / 7)
                base["is_spring"] = int(3 <= base["month"] <= 5)
                base["is_summer"] = int(6 <= base["month"] <= 8)
                base["is_autumn"] = int(9 <= base["month"] <= 11)
                base["is_winter"] = int(base["month"] == 12 or base["month"] <= 2)
                base["summer_shutdown"] = int(base["month"] == 8)
                base["post_shutdown_recovery"] = int(base["month"] == 9)
                month_avg_val = float(month_avg_map.get(base["month"], month_avg_default))
                base["article_monthly_mean"] = month_avg_val
                base["seasonal_ratio"] = last_seasonal_ratio
                base["article_avg_monthly_sales"] = article_avg_sales
                base["article_sales_variance"] = article_sales_var

                # price-related features: propagate last known values
                if not hist.empty:
                    last_row = hist.iloc[-1]
                    for col in self.PRICE_COLUMNS:
                        if col in hist.columns:
                            try:
                                base[col] = float(last_row.get(col, 0.0))
                            except Exception:
                                base[col] = 0.0
                            base[f"{col}_change_1"] = 0.0
                            base[f"{col}_pct_change"] = 0.0
                else:
                    for col in self.PRICE_COLUMNS:
                        base[col] = 0.0
                        base[f"{col}_change_1"] = 0.0
                        base[f"{col}_pct_change"] = 0.0

                # lag
                tgt_hist = hist[target_col].astype(float)
                for lag in lag_list:
                    base[f"{target_col}_lag_{lag}"] = (
                        float(tgt_hist.iloc[-lag]) if len(tgt_hist) >= lag else 0.0
                    )
                # rolling on last window values (equivalent to shift(1) at train time)
                for window in roll_windows:
                    tail = tgt_hist.tail(window)
                    arr = tail.values.astype(float)
                    base[f"{target_col}_roll_mean_{window}"] = (
                        float(arr.mean()) if arr.size > 0 else 0.0
                    )
                    base[f"{target_col}_roll_std_{window}"] = (
                        float(arr.std(ddof=0)) if arr.size > 0 else 0.0
                    )
                    base[f"{target_col}_roll_med_{window}"] = (
                        float(np.median(arr)) if arr.size > 0 else 0.0
                    )

                # rate of change
                if len(tgt_hist) >= 2:
                    prev = float(tgt_hist.iloc[-1])
                    prev1 = float(tgt_hist.iloc[-2])
                    base["roc_1"] = (prev - prev1) / (abs(prev1) + 1e-7)
                else:
                    base["roc_1"] = 0.0
                if len(tgt_hist) >= 3:
                    prev1 = float(tgt_hist.iloc[-2])
                    prev2 = float(tgt_hist.iloc[-3])
                    base["roc_2"] = (prev1 - prev2) / (abs(prev2) + 1e-7)
                else:
                    base["roc_2"] = 0.0

                all_rows.append(base)
                last_month_sales_flag = base["sold_this_month"]

                # Append placeholder, verrà riempito dalla predizione autoregressiva
                append_row = base.copy()
                append_row[target_col] = np.nan
                hist = pd.concat(
                    [hist, pd.DataFrame([append_row])],
                    ignore_index=True,
                )
                hist[date_col] = pd.to_datetime(hist[date_col], errors="coerce")
                hist = hist.dropna(subset=[date_col]).sort_values(date_col).reset_index(
                    drop=True
                )
                hist_month_series = hist[date_col].dt.month
                month_avg_map = (
                    hist.assign(__month_tmp=hist_month_series)
                    .groupby("__month_tmp")[target_col]
                    .mean()
                    .to_dict()
                )
                month_avg_default = (
                    float(hist[target_col].astype(float).mean())
                    if not hist.empty
                    else 0.0
                )
                article_avg_sales = month_avg_default
                article_sales_var = (
                    float(hist[target_col].astype(float).std(ddof=0))
                    if len(hist) > 1
                    else 0.0
                )
                if "seasonal_ratio" in hist.columns:
                    ratio_series = hist["seasonal_ratio"].dropna()
                    if not ratio_series.empty:
                        last_seasonal_ratio = float(ratio_series.iloc[-1])

        future_df = pd.DataFrame(all_rows)
        # riordina colonne come nelle feature attese
        for col in feature_columns:
            if col not in future_df.columns:
                future_df[col] = 0.0
        return future_df

    # ----------------- API: training (con split) -----------------
    def prepare_and_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
        df = df.copy()
        df = self.__add_temporal_features(df)
        df = self.__add_activity_features(df)
        df = self.__add_lag_features(df)
        df = self.__add_rolling_features(df)
        df = self.__add_seasonal_features(df)
        df = self.__add_monthly_baseline_features(df)

        if self.__config.target_col not in df.columns:
            raise ValueError(
                f"Target column {self.__config.target_col} not in dataframe."
            )

        y = df[self.__config.target_col]
        feature_columns = self.get_feature_columns(df)
        X = df[feature_columns].fillna(0)

        sorted_idx = df.sort_values(self.__config.date_col).index
        X_sorted = X.loc[sorted_idx]
        y_sorted = y.loc[sorted_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sorted, y_sorted, test_size=test_size, shuffle=False
        )

        return cast(
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]],
            (X_train, y_train, X_test, y_test, feature_columns),
        )

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.__add_temporal_features(df)
        df = self.__add_activity_features(df)
        df = self.__add_lag_features(df)
        df = self.__add_rolling_features(df)
        df = self.__add_seasonal_features(df)
        df = self.__add_monthly_baseline_features(df)
        df = self.__add_price_features(df)
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        feature_columns = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in {self.__config.target_col, self.__config.date_col, "article_id"}
        ]
        if getattr(self.__config, "exclude", None):
            feature_columns = [
                c for c in feature_columns if c not in set(self.__config.exclude)
            ]
        return feature_columns
