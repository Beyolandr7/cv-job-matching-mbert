from typing import Optional

from pydantic import BaseModel


class UserLogin(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str


class UserResponse(BaseModel):
    id: str = "admin-001"
    username: str = "admin"
    email: str = "admin@skripsi.local"
    full_name: str = "Administrator (Validator Ahli)"
    role: str = "admin"


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TokenPayload(BaseModel):
    sub: Optional[str] = None
