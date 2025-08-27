from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "LinkedIn Scraper API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # LinkedIn credentials
    LINKEDIN_EMAIL: str = "ibrahim.adeshina.10@gmail.com"
    LINKEDIN_PASSWORD: str = "@MisterSinister1"
    
    # File storage
    STORAGE_PATH: str = "./storage"
    DOWNLOADS_PATH: str = "./storage/downloads"
    SCREENSHOTS_PATH: str = "./storage/screenshots"
    PROCESSED_PATH: str = "./storage/processed"
    
    # Scraping settings
    MAX_PROFILES_PER_REQUEST: int = 50
    SCRAPING_TIMEOUT: int = 300
    PLAYWRIGHT_HEADLESS: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()
