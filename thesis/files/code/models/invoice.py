class Invoice(BaseModel):
    id: int
    customer: str
    creating_date: date
    expiring_date: Optional[date] = None
    expected_shipping_date: Optional[date] = None
    real_shipping_date: Optional[date] = None
    total: float
    articles: List[ArticleInvoice]
