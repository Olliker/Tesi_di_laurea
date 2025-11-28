class DataFrameService:
    """Versione ridotta per la tesi: mostra solo i passaggi principali."""

    PRICE_COLUMNS = [
        "unit_price_discounted",
        "unit_price_list",
        "discount_percent",
        "total_before_discount",
        "total_after_discount",
        "vat_rate",
    ]

    def __init__(self):
        self.__config = DataFrameConfig()  # date_col, target_col, exclude

    def get_date_col(self) -> str:
        return self.__config.date_col

    def get_target_col(self) -> str:
        return self.__config.target_col

    def get_price_columns(self, include_derivatives: bool = False) -> list[str]:
        cols = list(self.PRICE_COLUMNS)
        if include_derivatives:
            cols.extend(f"{c}{s}" for c in self.PRICE_COLUMNS for s in ("_change_1", "_pct_change"))
        return cols

    def prepare_and_split(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
        # Pipeline sintetica: verifica target, selezione feature numeriche, split temporale
        if self.__config.target_col not in df.columns:
            raise ValueError(f"Target column {self.__config.target_col} not in dataframe.")
        y = df[self.__config.target_col]
        feature_columns = self.get_feature_columns(df)
        X = df[feature_columns].fillna(0)

        sorted_idx = df.sort_values(self.__config.date_col).index
        X_train, X_test, y_train, y_test = train_test_split(
            X.loc[sorted_idx], y.loc[sorted_idx], test_size=test_size, shuffle=False
        )
        return X_train, y_train, X_test, y_test, feature_columns

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.__config.target_col not in df.columns:
            raise ValueError(f"Target column {self.__config.target_col} not in dataframe.")
        feature_columns = self.get_feature_columns(df)
        return df[feature_columns].fillna(0)

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        feature_columns = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in {self.__config.target_col, self.__config.date_col, "article_id"}
        ]
        if getattr(self.__config, "exclude", None):
            feature_columns = [c for c in feature_columns if c not in set(self.__config.exclude)]
        return feature_columns
