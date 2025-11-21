# app/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Configuracion base 

@dataclass
class Settings:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    spotify_client_id: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    spotify_client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    tmdb_api_key: str = os.getenv("TMDB_API_KEY", "")
    region: str = os.getenv("REGION", "AR")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

settings = Settings()

# Validaciones mínimas
if not settings.telegram_bot_token:
    raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el archivo .env")

if not settings.tmdb_api_key:
    raise RuntimeError("Falta TMDB_API_KEY en el archivo .env")

if not settings.groq_api_key:
    raise RuntimeError("Falta GROQ_API_KEY en el archivo .env")


# Cliente Groq listo para usar

groq_client = Groq(api_key=settings.groq_api_key)

# Constantes TMDB
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_LANG = "es-AR"

