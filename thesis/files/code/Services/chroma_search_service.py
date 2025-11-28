class ChromaSearchService:
    def __init__(self, repo: Optional[ChromaRepository] = None):
        self._repo = repo or ChromaRepository()

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _fallback_substring_search(
        self, query_text: str, n_results: int
    ) -> ForecastSearchResult:
        try:
            manual_col = self._repo.get_collection("forecast_predictions")
            if not manual_col:
                return ForecastSearchResult(
                    predictions=[], explanations=[], count=0, fallback=True
                )
            raw = manual_col.get(include=["documents", "metadatas"])
            if not raw:
                return ForecastSearchResult(
                    predictions=[], explanations=[], count=0, fallback=True
                )
            docs = cast(List[List[str]], raw.get("documents", []) or [])
            metas = cast(List[List[Dict[str, Any]]], raw.get("metadatas", []) or [])
            if not docs or not docs[0]:
                return ForecastSearchResult(
                    predictions=[], explanations=[], count=0, fallback=True
                )
            q_norm = self._normalize(query_text)
            matches: List[tuple[str, Dict[str, Any]]] = []
            for i, doc in enumerate(docs[0]):
                d_norm = self._normalize(doc)
                if all(tok in d_norm.split() for tok in q_norm.split()):
                    meta = (
                        metas[0][i] if metas and metas[0] and i < len(metas[0]) else {}
                    )
                    matches.append((doc, meta))
            if not matches:
                return ForecastSearchResult(
                    predictions=[], explanations=[], count=0, fallback=True
                )
            matches = matches[:n_results]
            print(
                f"[chroma] normalized fallback matched {len(matches)} docs", flush=True
            )
            preds = [m[0] for m in matches]
            expls: List[Optional[str]] = [None for _ in matches]
            return ForecastSearchResult(
                predictions=preds, explanations=expls, count=len(matches), fallback=True
            )
        except Exception as e:
            print(f"[chroma] normalized fallback error: {e}", flush=True)
            return ForecastSearchResult(
                predictions=[], explanations=[], count=0, fallback=True, error=str(e)
            )

    def query_forecast(
        self, query_text: str, n_results: int = 5, forecast_id: str = None
    ) -> ForecastSearchResult:
        result = self._repo.query_forecast(
            query_text=query_text, n_results=n_results, forecast_id=forecast_id
        )
        # result è un dict con chiavi: predictions, explanations, count, (eventualmente fallback, error)
        return ForecastSearchResult(
            predictions=result.get("predictions", []),
            explanations=result.get("explanations", []),
            count=result.get("count", 0),
            fallback=result.get("fallback", False),
            error=result.get("error") if "error" in result else None,
        )

    def get_context(
        self, question: str, result_count: int = 5, forecast_id: str = None
    ) -> ForecastSearchResult:
        return self.query_forecast(
            query_text=question, n_results=result_count, forecast_id=forecast_id
        )
