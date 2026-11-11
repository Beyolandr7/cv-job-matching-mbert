from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from core.config import settings
from core.security import create_access_token
from routers.deps import _build_admin_user, get_current_user
from schemas.auth import Token, UserLogin, UserResponse

router = APIRouter()


@router.post("/login", response_model=Token)
def login_user(user_in: UserLogin) -> dict:
    """
    Authenticates Admin user using hardcoded credentials (as specified in supervisor whiteboard)
    and returns a signed JWT access token without requiring a database server.
    """
    identifier = (user_in.username or user_in.email or "").strip()
    valid_identifiers = {
        settings.ADMIN_USERNAME.lower(),
        f"{settings.ADMIN_USERNAME.lower()}@skripsi.local",
        "admin@admin.com",
    }

    if identifier.lower() not in valid_identifiers or user_in.password != settings.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username atau password Admin tidak sesuai.",
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=settings.ADMIN_USERNAME,
        expires_delta=access_token_expires,
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": _build_admin_user(),
    }


@router.post("/token", response_model=Token, include_in_schema=False)
def login_swagger(form_data: OAuth2PasswordRequestForm = Depends()) -> dict:
    """OAuth2 password request form endpoint for Swagger UI documentation."""
    if form_data.username.strip().lower() != settings.ADMIN_USERNAME.lower() or form_data.password != settings.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    access_token = create_access_token(subject=settings.ADMIN_USERNAME)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": _build_admin_user(),
    }


@router.get("/me", response_model=UserResponse)
def read_current_user(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Returns profile information of the currently authenticated Admin."""
    return current_user
