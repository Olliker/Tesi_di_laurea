chromadb = APIRouter(dependencies=[Depends(jwt_dependencies.check_token)])
_search_service = None

def _get_service() -> ChromaSearchService:
    global _search_service
    if _search_service is None:
        print("[context] Initializing ChromaSearchService singleton", flush=True)
        _search_service = ChromaSearchService()
    return _search_service

@chromadb.get("/context")
async def get_context(question: str, result_count: int = 5, forecast_id: str = None):
    try:
        service = _get_service()
        resp = service.query_forecast(question, result_count, forecast_id)
        print(
            f"[context] query='{question}' forecast_id={forecast_id} results_pred={len(resp.predictions)}",
            flush=True,
        )
        return resp
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Context retrieval error: {e}")
