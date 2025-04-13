from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_API_BASE_URL: str='https://api.openai.com/v1'
    OPENAI_MODEL_NAME: str='gpt-3.5-turbo'
    LABEL_STUDIO_API_KEY: str
    LABEL_STUDIO_URL: str='http://localhost:8080'
    env: str = 'development'

    class Config:
        env_file = '.env'

settings = Settings()
