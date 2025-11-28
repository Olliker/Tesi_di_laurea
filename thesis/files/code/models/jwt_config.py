@dataclass
class JWTConfig:
    secret_key: str = os.getenv("JWT_SECRET_KEY", "default")
    algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
