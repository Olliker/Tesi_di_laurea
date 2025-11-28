@dataclass
class XGBoostConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 6
    n_jobs: int = -1
    min_child_weight: float = 5.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    random_state: int = 42
    tree_method: str = "hist"
    max_bin: int = 256
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"
    early_stopping_rounds: int = 50
    target_transform: str = "log1p"
    verbosity: int = 1
