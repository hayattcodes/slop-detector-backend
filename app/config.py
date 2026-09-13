"""
Central place for all configuration. Everything is read from environment
variables (via a .env file) so nothing sensitive is hardcoded in the code.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    secret_key: str = "dev-only-insecure-key-change-me"
    access_token_expire_minutes: int = 10080  # 7 days
    database_url: str = "sqlite:///./detector.db"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,https://the-slop-detector.vercel.app"

    # Optional heavy-engine backend (Cornell server). Blank = feature disabled.
    deep_scan_api_url: str = ""
    deep_scan_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def deep_scan_enabled(self) -> bool:
        return bool(self.deep_scan_api_url)


settings = Settings()
