from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union

from jose import jwt

from core.config import settings


def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Creates an encoded JWT access token with an expiration timestamp."""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return jwt.encode({"exp": expire, "sub": str(subject)}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
