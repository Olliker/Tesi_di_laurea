class InvoiceConversionService:
    def __init__(self):
        self.duck: DuckDBRepository = DuckDBRepository()

    def convert_db_data_to_df(self) -> pd.DataFrame:
        df = self.duck.fetch_fact_invoices()

        date_columns = [
            "invoice_date",
            "expiring_date",
            "shipping_date",
            "expected_shipping_date",
        ]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        numeric_columns = [
            "article_id",
            "customer_id",
            "is_italy",
            "customer_avg_quantity",
            "customer_order_frequency",
            "is_top_customer",
            "quantity",
            "unit_price_discounted",
            "unit_price_list",
            "discount_percent",
            "total_before_discount",
            "total_after_discount",
            "vat_rate",
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        sum_columns = ["quantity"]

        raw_ordered_columns: List[str] = [
            "article_id",
            "article_name",
            "invoice_id",
            "customer_id",
            "invoice_date",
            "expiring_date",
            "shipping_date",
            "expected_shipping_date",
            "quantity",
            "total_before_discount",
            "total_after_discount",
        ]
        aggregated_ordered_columns: List[str] = [
            "article_id",
            "article_name",
            "invoice_month",
            "quantity",
            "sold_this_month",
        ]

        aggregated = False
        should_aggregate = "invoice_date" in df.columns and "article_id" in df.columns
        if should_aggregate:
            df = df.dropna(subset=["invoice_date"])
            if not df.empty:
                df = df.copy()
                df["invoice_month"] = (
                    df["invoice_date"].dt.to_period("M").dt.to_timestamp()
                )
                group_cols = [
                    col
                    for col in ["article_id", "article_name", "invoice_month"]
                    if col in df.columns
                ]
                agg_map: dict[str, str] = {}
                for col in sum_columns:
                    if col in df.columns:
                        agg_map[col] = "sum"
                if agg_map and group_cols:
                    df = (
                        df.groupby(group_cols, dropna=False)
                        .agg(agg_map)
                        .reset_index()
                    )
                    aggregated = True
                elif group_cols:
                    df = df[group_cols].drop_duplicates().reset_index(drop=True)
                    aggregated = True

        if aggregated and not df.empty:
            grouped_frames: list[pd.DataFrame] = []
            for aid, grp in df.sort_values(["article_id", "invoice_month"]).groupby(
                "article_id", sort=False
            ):
                grp = grp.sort_values("invoice_month").copy()
                start = grp["invoice_month"].min()
                end = grp["invoice_month"].max()
                if pd.isna(start) or pd.isna(end):
                    continue
                full_range = pd.date_range(start, end, freq="MS")
                resampled = grp.set_index("invoice_month").reindex(full_range)
                resampled["article_id"] = aid
                if "article_name" in resampled.columns:
                    resampled["article_name"] = (
                        resampled["article_name"].ffill().bfill().fillna(
                            grp["article_name"].dropna().iloc[0]
                            if "article_name" in grp.columns
                            and not grp["article_name"].dropna().empty
                            else f"Articolo_{aid}"
                        )
                    )
                resampled["quantity"] = resampled["quantity"].fillna(0.0)
                resampled["sold_this_month"] = (
                    resampled["quantity"].fillna(0.0) > 0
                ).astype(int)
                resampled = resampled.reset_index().rename(
                    columns={"index": "invoice_month"}
                )
                grouped_frames.append(resampled)

            if grouped_frames:
                df = pd.concat(grouped_frames, ignore_index=True)
            else:
                df = df.iloc[0:0]

        if aggregated:
            drop_columns = [
                "invoice_date",
                "expiring_date",
                "shipping_date",
                "expected_shipping_date",
                "invoice_id",
                "customer_id",
                "total_before_discount",
                "total_after_discount",
            ]
            existing_drop = [c for c in drop_columns if c in df.columns]
            if existing_drop:
                df = df.drop(columns=existing_drop)

        ordered_columns = (
            aggregated_ordered_columns if aggregated else raw_ordered_columns
        )
        existing: List[str] = [c for c in ordered_columns if c in df.columns]
        return cast(pd.DataFrame, df[existing]) if existing else df
