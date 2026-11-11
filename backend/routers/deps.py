from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from core.config import settings
from schemas.auth import TokenPayload, UserResponse

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/token",
    auto_error=False,
)


def _build_admin_user() -> UserResponse:
    return UserResponse(
        id="admin-001",
        username=settings.ADMIN_USERNAME,
        email=f"{settings.ADMIN_USERNAME}@skripsi.local",
        full_name="Administrator (Validator)",
        role="admin",
    )


def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[UserResponse]:
    """
    Returns the authenticated Admin UserResponse if a valid JWT is provided,
    otherwise returns None without throwing 401 (User Biasa).
    """
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        token_data = TokenPayload(**payload)
        if token_data.sub == settings.ADMIN_USERNAME or token_data.sub == "admin-001":
            return _build_admin_user()
    except (JWTError, Exception):
        return None
    return None


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> UserResponse:
    """
    Strictly validates the JWT token and returns the authenticated Admin UserResponse.
    Throws 401 Unauthorized if invalid or missing.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Sesi Admin tidak valid atau sudah berakhir. Silakan login kembali.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        token_data = TokenPayload(**payload)
        if token_data.sub is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    if token_data.sub not in (settings.ADMIN_USERNAME, "admin-001"):
        raise credentials_exception

    return _build_admin_user()
