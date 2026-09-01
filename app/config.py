from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("./data")
    embedding_dim: int = 384
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    azure_storage_connection_string: str = ""
    azure_blob_container: str = "document-assets"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def db_path(self) -> Path:
        return self.data_dir / "intelligence.db"

    @property
    def blob_dir(self) -> Path:
        return self.data_dir / "blobs"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def seed_dir(self) -> Path:
        return self.data_dir / "seed"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.blob_dir, self.upload_dir, self.seed_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
