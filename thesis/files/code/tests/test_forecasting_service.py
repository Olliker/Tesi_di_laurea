class FakeXGB:
    def __init__(self):
        self.pred_called = None
        self.model = object()

    def predict(self, *, start_date, periods):
        self.pred_called = (start_date, periods)
        results = SimpleNamespace(
            future_predictions={
                "prediction_date": {0: "2024-02-01", 1: "2024-02-08"},
                "y": {0: 1, 1: 2},
            },
            other="ok",
        )
        X_pred = [[0.1, 0.2]]
        features = ["f1", "f2"]
        return results, X_pred, features

    def get_model(self):
        return self.model


class FakeShap:
    def __init__(self):
        self.called = None

    def explain_prediction(self, X, model):
        self.called = (tuple(map(tuple, X)), model)
        return SimpleNamespace(values=[[0.1, -0.1]])


class FakeNormalize:
    def __init__(self):
        self.pred_in = None
        self.expl_in = None
        self.ctx_called = False

    def translate_multiple_predictions_to_text(self, results):
        self.pred_in = results
        return ["p1", "p2"]

    def translate_multiple_explanations_to_text(self, shap_expl, feature_cols, results):
        self.expl_in = (shap_expl, tuple(feature_cols), results)
        return ["e1"]

    def build_embedding_documents(self, prediction_result, shap_expl, feature_cols):
        self.ctx_called = True
        return ["doc"], [{"article_id": "10"}]


class FakeChromaRepo:
    pass


class FakeDBRepo:
    pass


class FakePersistence:
    def __init__(self):
        self.saved = None

    def save(
        self,
        *,
        translated_predictions,
        translated_explanations,
        context_documents,
        context_metadatas,
        forecast_id,
        db_payload,
    ):
        self.saved = (
            translated_predictions,
            translated_explanations,
            context_documents,
            context_metadatas,
            forecast_id,
            db_payload,
        )


def make_service():
    fs = ForecastingService(
        xgb=cast(XGBoostService, FakeXGB()),
        shap=cast(ShapService, FakeShap()),
        chroma_repo=cast(ChromaRepository, FakeChromaRepo()),
        db_repo=cast(DBRepository, FakeDBRepo()),
        normalize=cast(DataNormalizeService, FakeNormalize()),
    )
    setattr(fs, "_ForecastingService__persistence", FakePersistence())
    return fs


def test_forecast_happy_path_builds_payload_and_persists():
    fs = make_service()
    out = fs.forecast(
        forecast_id="fid",
        start_date=datetime(2024, 2, 1),
        periods=2,
    )
    xgb = getattr(fs, "_ForecastingService__xgboost_service")
    assert xgb.pred_called == (datetime(2024, 2, 1), 2)
    shap = getattr(fs, "_ForecastingService__shap_service")
    assert shap.called[1] is xgb.get_model()
    norm = getattr(fs, "_ForecastingService__normalize_service")
    assert norm.pred_in is not None and norm.expl_in is not None and norm.ctx_called
    saved = getattr(fs, "_ForecastingService__persistence").saved
    assert saved is not None and saved[4] == "fid"
    assert saved[0] == ["p1", "p2"] and saved[1] == ["e1"]
    assert saved[2] == ["doc"] and saved[3] == [{"article_id": "10"}]
    assert out["forecast_id"] == "fid"
    assert out["start_date"].startswith("2024-02-01")
    assert out["frequency"] == "2W"
    assert out["periods"] == 2
    assert out["predictions"].future_predictions == {
        "prediction_date": {0: "2024-02-01", 1: "2024-02-08"},
        "y": {0: 1, 1: 2},
    }
    rows = out.get("predictions_rows")
    assert rows and rows[0]["prediction_date"].startswith("2024-02-01")
    assert rows[1]["y"] == 2
    assert out.get("end_date").startswith("2024-02-08")


def test_forecast_handles_df_build_error(monkeypatch):
    fs = make_service()

    def boom(*a, **k):
        raise ValueError("bad")

    monkeypatch.setattr("pandas.DataFrame.from_dict", boom)
    out = fs.forecast(
        forecast_id="fid",
        start_date=datetime(2024, 2, 1),
        periods=1,
    )
    assert "predictions_rows" not in out
