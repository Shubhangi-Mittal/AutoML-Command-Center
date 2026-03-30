from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	DATABASE_URL: str = "postgresql://automl:automl_secret@localhost:5432/automl"
	REDIS_URL: str = "redis://localhost:6379/0"
	MLFLOW_TRACKING_URI: str = "http://localhost:5000"
	ANTHROPIC_API_KEY: str = ""
	GROQ_API_KEY: str = ""
	UPLOAD_DIR: str = "./uploads"
	MODEL_DIR: str = "./models"
	MAX_UPLOAD_SIZE_MB: int = 100
	ALLOWED_ORIGINS: str = "http://localhost:3000"

	model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
