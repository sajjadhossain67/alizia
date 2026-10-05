"""Alizia AI Backend - Core Configuration (Pydantic v2)"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ALIZIA_", env_nested_delimiter="__")
    
    # Application
    APP_NAME: str = "Alizia AI Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Security
    SECRET_KEY: str = Field(default="alizia-dev-secret-key-must-change-in-production")
    API_SECRET_KEY: str = Field(default="alizia-dev-api-key-must-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    ALLOWED_HOSTS: list = ["localhost", "127.0.0.1", "0.0.0.0", "testserver", "*"]
    
    # Database
    POSTGRES_URL: str = Field(default="postgresql://localhost:5432/alizia")
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_MAX_OVERFLOW: int = 30
    
    # Model gateway
    DEFAULT_MODEL: str = "alizia-nova"
    FALLBACK_MODEL: str = "alizia-pulse"
    SUPPORTED_MODELS: list = [
        "alizia-nova", "alizia-pulse", "alizia-forge", 
        "alizia-vision", "alizia-edge", "alizia-embed-v1"
    ]
    
    # Token accounting
    INPUT_TOKEN_COST: float = 0.001
    OUTPUT_TOKEN_COST: float = 0.002
    REASONING_TOKEN_COST: float = 0.003
    EMBEDDING_TOKEN_COST: float = 0.0001
    
    # Rate limiting
    RATE_LIMIT_DEFAULT: int = 100
    RATE_LIMIT_BURST: int = 20
    RATE_LIMIT_PER_MINUTE: int = 1000
    RATE_LIMIT_PER_DAY: int = 100000
    
    # Safety
    SAFETY_POLICY_PATH: str = "config/safety_policy.json"
    MODERATION_ENABLED: bool = True
    
    # File storage
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_UPLOAD_EXTENSIONS: list = [
        ".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".txt", 
        ".md", ".html", ".json", ".xml", ".py", ".js", ".ts"
    ]
    
    # Agent defaults
    DEFAULT_MAX_STEPS: int = 100
    DEFAULT_MAX_TOKENS: int = 4096
    DEFAULT_TIMEOUT_SECONDS: int = 300
    
    # Observability
    SENTRY_DSN: str = ""
    OTEL_EXPORT_ENDPOINT: str = ""
    
    # Feature flags
    ENABLE_AGENTS: bool = True
    ENABLE_RAG: bool = True
    ENABLE_VISION: bool = True
    ENABLE_CODE_EXECUTION: bool = True
    EMBEDDING_SERVICE_ENABLED: bool = True
    
    # Model pricing (per 1K tokens)
    PRICING: dict = {
        "alizia-nova": {"input": 0.001, "output": 0.002, "reasoning": 0.003},
        "alizia-pulse": {"input": 0.0002, "output": 0.0005, "reasoning": 0.001},
        "alizia-forge": {"input": 0.0015, "output": 0.003, "reasoning": 0.005},
        "alizia-vision": {"input": 0.002, "output": 0.004, "reasoning": 0.006},
        "alizia-edge": {"input": 0.0001, "output": 0.0002, "reasoning": 0.0005},
        "alizia-embed-v1": {"input": 0.0001, "output": 0.0},
    }
    
    @property
    def rate_limit_default(self) -> int:
        return self.RATE_LIMIT_DEFAULT
    
    @property
    def rate_limit_burst(self) -> int:
        return self.RATE_LIMIT_BURST


settings = Settings()