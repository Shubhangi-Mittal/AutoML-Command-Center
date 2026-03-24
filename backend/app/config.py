from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	DATABASE_URL: str = "postgresql://automl:automl_secret@localhost:5432/automl"
	REDIS_URL: str = "redis://localhost:6379/0"
	MLFLOW_TRACKING_URI: str = "http://localhost:5000"
	ANTHROPIC_API_KEY: str = "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"
	UPLOAD_DIR: str = "./uploads"
	MODEL_DIR: str = "./models"
	MAX_UPLOAD_SIZE_MB: int = 100

	model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
