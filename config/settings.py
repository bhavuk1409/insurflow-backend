"""
Configuration management using Pydantic Settings.
Loads environment variables and provides typed configuration.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    groq_api_key: str
    
    # Service URLs
    damage_detection_url: str = "http://localhost:8000"
    
    # OCR Configuration
    ocr_engine: str = "paddleocr"
    ocr_lang: str = "en"
    
    # LLM Configuration
    llm_model: str = "mixtral-8x7b-32768"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    
    # Application Settings
    log_level: str = "INFO"
    max_file_size_mb: int = 10
    
    # Policy Configuration
    default_max_payout: float = 500000.0
    default_currency: str = "INR"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
