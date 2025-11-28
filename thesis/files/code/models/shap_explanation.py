class ShapExplanation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: np.ndarray[Any, np.dtype[np.float64]]

    @field_serializer("values")
    def _ser_np(self, v: np.ndarray[Any, np.dtype[np.float64]]):
        if v is not None:
            return v.tolist()
        return None