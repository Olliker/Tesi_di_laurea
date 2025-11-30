@pytest.mark.asyncio
async def test_chromadb_context_success(monkeypatch, client, app):

    class FakeService:
        def query_forecast(
            self, question: str, n_results: int = 5, forecast_id: str | None = None
        ):
            return SimpleNamespace(
                predictions=["p1", "p2"],
                explanations=[None, None],
                count=2,
                fallback=False,
            )

    app.dependency_overrides[jwt_deps.jwt_dependencies.check_token] = lambda: {
        "sub": "test"
    }
    monkeypatch.setattr(chroma_api, "_get_service", lambda: FakeService())

    r = await client.get("/context", params={"question": "abc", "result_count": 2})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2 and data["predictions"] == ["p1", "p2"]

    app.dependency_overrides.pop(jwt_deps.jwt_dependencies.check_token, None)


@pytest.mark.asyncio
async def test_chromadb_context_error(monkeypatch, client, app):

    class BoomService:
        def query_forecast(self, *a, **k):
            raise RuntimeError("boom")

    app.dependency_overrides[jwt_deps.jwt_dependencies.check_token] = lambda: {
        "sub": "test"
    }
    monkeypatch.setattr(chroma_api, "_get_service", lambda: BoomService())

    r = await client.get("/context", params={"question": "abc"})
    assert r.status_code == 500


@pytest.mark.asyncio
async def test_chromadb_context_with_forecast_id(monkeypatch, client, app):
    
    class FakeService:
        def __init__(self):
            self.last = None

        def query_forecast(
            self, question: str, n_results: int = 5, forecast_id: str | None = None
        ):
            self.last = (question, n_results, forecast_id)
            return SimpleNamespace(
                predictions=[], explanations=[], count=0, fallback=False
            )

    svc = FakeService()
    app.dependency_overrides[jwt_deps.jwt_dependencies.check_token] = lambda: {
        "sub": "test"
    }
    monkeypatch.setattr(chroma_api, "_get_service", lambda: svc)

    fid = "abc-123"
    r = await client.get(
        "/context", params={"question": "q", "result_count": 1, "forecast_id": fid}
    )
    assert r.status_code == 200
    assert svc.last == ("q", 1, fid)
    app.dependency_overrides.pop(jwt_deps.jwt_dependencies.check_token, None)

    app.dependency_overrides.pop(jwt_deps.jwt_dependencies.check_token, None)
