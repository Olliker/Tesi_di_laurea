authentication = APIRouter()


@authentication.get("/.well-known/jwks.json")
async def get_public_keys() -> JWKS:
    return JWKService().get_jwks()


@authentication.post("/token")
async def get_token(token: JWT = Depends(jwt_dependencies.get_new_token)) -> JWT:
    return token
