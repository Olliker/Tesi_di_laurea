class ForecastingService:
    def __init__(
        self,
        *,
        xgb: XGBoostService | None = None,
        shap: ShapService | None = None,
        chroma_repo: ChromaRepository | None = None,
        db_repo: DBRepository | None = None,
        normalize: DataNormalizeService | None = None,
    ):
        self.__xgboost_service = xgb or XGBoostService()
        self.__shap_service = shap or ShapService()
        self.__chroma_repository = chroma_repo or ChromaRepository()
        self.__db_repository = db_repo or DBRepository()
        self.__normalize_service: DataNormalizeService = (
            normalize or DataNormalizeService()
        )
        self.__persistence = ForecastPersistenceService(
            chroma_repo=self.__chroma_repository, db_repo=self.__db_repository
        )

    def forecast(
        self,
        *,
        forecast_id: str,
        start_date: datetime,
        periods: int,
    ) -> Dict[str, Any]:
        results, X_pred, feature_columns = self.__xgboost_service.predict(
            start_date=start_date, periods=periods
        )


        shap_explanation = self.__shap_service.explain_prediction(
            X_pred, self.__xgboost_service.get_model()
       )

        normalizer = self.__normalize_service

        translated_predictions = normalizer.translate_multiple_predictions_to_text(
            results
        )

        translated_explanations = normalizer.translate_multiple_explanations_to_text(
            shap_explanation, feature_columns, results
        )

        context_documents, context_metadatas = normalizer.build_embedding_documents(
            results, shap_explanation, feature_columns
        )

        data: Dict[str, Any] = {
            "forecast_id": forecast_id,
            "predictions": results,
            "start_date": start_date.isoformat(),
            "frequency": "M",
            "periods": periods,
        }

        df = pd.DataFrame.from_dict(results.future_predictions)
        data["predictions_df"] = df.to_dict(orient="records")

        self.__persistence.save(
            translated_predictions=translated_predictions,
            translated_explanations=translated_explanations,
            context_documents=context_documents,
            context_metadatas=context_metadatas,
            forecast_id=forecast_id,
            db_payload=data,
        )

        return data
