import os
from typing import Any

import jwt
from jwt import PyJWKClient


class JWTService:
    def verify_jwt(self, token: str) -> Any:
        jwks_client = PyJWKClient(
            uri=str(
                os.getenv(
                    "JWK_CLIENT_URL",
                    "http://auth-server-service:8002/.well-known/jwks.json",
                )
            )
        )
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(jwt=token, key=signing_key, algorithms=["RS256"])
