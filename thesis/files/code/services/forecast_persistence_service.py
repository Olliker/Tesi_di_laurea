from typing import Any, Dict

from forecasting.app.repositories.chroma_repository import ChromaRepository
from forecasting.app.repositories.db_repository import DBRepository


class ForecastPersistenceService:
    def __init__(
        self,
        *,
        chroma_repo: ChromaRepository | None = None,
        db_repo: DBRepository | None = None,
    ):
        self._chroma = chroma_repo or ChromaRepository()
        self._db = db_repo or DBRepository()

    def save(
        self,
        *,
        translated_predictions: list[str],
        translated_explanations: list[str],
        context_documents: list[str] | None = None,
        context_metadatas: list[Dict[str, Any]] | None = None,
        forecast_id: str,
        db_payload: Dict[str, Any],
    ) -> None:
        self._chroma.save_translated_data_to_chroma(
            translated_predictions,
            translated_explanations,
            forecast_id,
            context_documents=context_documents,
            context_metadatas=context_metadatas,
        )
        self._db.save_predictions(db_payload)