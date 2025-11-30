jwt = APIRouter()

@jwt.post("/verify-token")
def token_verification(payload: dict = Depends(jwt_dependencies.check_token)):
    return {"msg": "token ok"}
