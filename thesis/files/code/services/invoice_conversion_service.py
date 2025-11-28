class InvoiceConversionService:
    """Versione ridotta per la tesi: mostra solo i passaggi chiave."""

    def __init__(self):
        self.duck: DuckDBRepository = DuckDBRepository()

    def convert_db_data_to_df(self) -> pd.DataFrame:
        df = self.duck.fetch_fact_invoices()

        # Normalizzazione colonne data e numeriche
        for col in ("invoice_date", "expiring_date", "shipping_date", "expected_shipping_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in ("article_id", "customer_id", "quantity", "unit_price_discounted", "unit_price_list"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Aggregazione per mese/articolo sommando le quantità
        if {"invoice_date", "article_id"} <= set(df.columns):
            df = df.dropna(subset=["invoice_date"]).copy()
            df["invoice_month"] = df["invoice_date"].dt.to_period("M").dt.to_timestamp()
            df = (
                df.groupby(["article_id", "article_name", "invoice_month"], dropna=False)
                .agg({"quantity": "sum"})
                .reset_index()
            )

            # Riempimento buchi temporali e flag vendite mensili
            filled: list[pd.DataFrame] = []
            for aid, grp in df.groupby("article_id", sort=False):
                rng = pd.date_range(grp["invoice_month"].min(), grp["invoice_month"].max(), freq="MS")
                res = grp.set_index("invoice_month").reindex(rng)
                res["article_id"] = aid
                res["article_name"] = res["article_name"].ffill().bfill().fillna(f"Articolo_{aid}")
                res["quantity"] = res["quantity"].fillna(0.0)
                res["sold_this_month"] = (res["quantity"] > 0).astype(int)
                filled.append(res.reset_index().rename(columns={"index": "invoice_month"}))
            df = pd.concat(filled, ignore_index=True) if filled else df

        # ...
        # (Altre colonne/filtri mantenuti nel codice completo non mostrato)
        # ...

        cols = ["article_id", "article_name", "invoice_month", "quantity", "sold_this_month"]
        existing = [c for c in cols if c in df.columns]
        return df[existing] if existing else df
