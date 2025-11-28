class NotificationService:
    def __init__(self):
        self._notify_env = "LARAVEL_NOTIFY_URL"

    def _get_token(self) -> Optional[str]:
        url = os.getenv("LARAVEL_LOGIN_URL")
        email = os.getenv("LARAVEL_MAIL")
        password = os.getenv("LARAVEL_PASSWORD")
        if not url or not email or not password:
            return None
        r = requests.post(url, json={"email": email, "password": password}, timeout=5)
        r.raise_for_status()
        data = (
            r.json()
            if r.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        return data.get("token")

    def notify(self, payload: NotificationEvent | dict[str, Any]) -> None:
        url = os.getenv(self._notify_env)
        if not url:
            return
        data = payload.model_dump() if isinstance(payload, NotificationEvent) else payload
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"} if token else None
        requests.post(url, json=data, headers=headers, timeout=5)
