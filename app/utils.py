# app/utils.py
from typing import Dict, Any, List
import json
from pathlib import Path
import requests
import time
from openai import OpenAI
from .config import settings
import logging

logger = logging.getLogger("moodflix")

HISTORY_PATH = Path("data/conversation_history.json")

# Control de rate limit
_LAST_OPENAI_CALL: float = 0.0
_OPENAI_CALLS_THIS_MINUTE: int = 0
_OPENAI_MINUTE_START: float = 0.0


def parse_user_intent_with_openai(text: str) -> Dict[str, Any]:
    """
    Parsea con FALLBACK INTELIGENTE (prioridad al fallback para evitar rate limits).
    Solo usa OpenAI si el fallback no está seguro.
    """
    global _LAST_OPENAI_CALL, _OPENAI_CALLS_THIS_MINUTE, _OPENAI_MINUTE_START
    
    text_lower = text.lower()

    # ===== FALLBACK MEJORADO (PRIORIDAD) =====
    def smart_fallback() -> Dict[str, Any]:
        tipo = "unknown"
        mood = None
        activity = None
        genre = None

        # --- DETECCIÓN DE TIPO ---
        # Música
        if any(w in text_lower for w in [
            "música", "musica", "canción", "cancion", "canciones", "tema", "temas", 
            "song", "playlist", "escuchar", "poneme", "tirame"
        ]):
            tipo = "music"
        
        # Películas
        elif any(w in text_lower for w in [
            "peli", "película", "pelicula", "peliculas", "películas", 
            "film", "filme", "movie", "ver una"
        ]):
            tipo = "movie"
        
        # Series
        elif any(w in text_lower for w in [
            "serie", "series", "capítulo", "capitulo", "temporada", 
            "temporadas", "maratonear", "bingewatching"
        ]):
            tipo = "series"
        
        # Si pide recomendación genérica
        elif any(w in text_lower for w in ["recomend", "recomiend", "pasame", "dame"]):
            # Intentar inferir del contexto
            if any(w in text_lower for w in ["ver", "mirar", "pantalla"]):
                tipo = "movie"  # default para ver
            else:
                tipo = "music"  # default general

        # --- DETECCIÓN DE MOOD ---
        mood_map = {
            "triste": ["triste", "bajón", "bajon", "depre", "mal", "down", "sad"],
            "estresada": ["estres", "estresad", "ansi", "nervios", "agobiad", "preocupad"],
            "relajada": ["relajad", "tranqui", "calm", "paz", "chill", "descansar"],
            "feliz": ["feliz", "content", "alegre", "bien", "happy", "genial"],
            "con energía": ["energía", "energia", "activ", "pilas", "motivad", "manija"],
            "cansada": ["cansad", "agotad", "sin energía", "muert", "revenid"],
            "nostálgica": ["nostálgi", "nostalgic", "recuerdos", "antes"],
            "enojada": ["enojad", "bronca", "rabia", "furioso", "angry"],
        }

        for mood_name, keywords in mood_map.items():
            if any(kw in text_lower for kw in keywords):
                mood = mood_name
                break

        # --- DETECCIÓN DE ACTIVIDAD ---
        activity_map = {
            "correr": ["correr", "running", "gym", "entrenar", "ejercicio", "deporte"],
            "estudiar": ["estudiar", "estudio", "trabajar", "trabajo", "focus", "concentrar"],
            "dormir": ["dormir", "acostar", "descansar", "sleep"],
            "cocinar": ["cocinar", "cocina", "cocinando", "preparar comida"],
            "limpiar": ["limpiar", "limpieza", "limpiar casa", "ordenar"],
            "viajar": ["viajar", "viaje", "ruta", "manejando", "conducir"],
        }

        for act_name, keywords in activity_map.items():
            if any(kw in text_lower for kw in keywords):
                activity = act_name
                break

        # --- DETECCIÓN DE GÉNERO ---
        genre_keywords = {
            "terror": ["terror", "miedo", "horror", "susto"],
            "comedia": ["comedia", "risa", "gracioso", "divertid", "humor"],
            "drama": ["drama", "dramático"],
            "romance": ["romance", "romántic", "amor"],
            "acción": ["acción", "accion", "action"],
            "ciencia ficción": ["ciencia ficción", "ciencia ficcion", "sci-fi", "scifi"],
            "pop": ["pop"],
            "rock": ["rock"],
            "reggaeton": ["reggaeton", "reguetón", "urbano"],
            "jazz": ["jazz"],
        }

        for genre_name, keywords in genre_keywords.items():
            if any(kw in text_lower for kw in keywords):
                genre = genre_name
                break

        return {
            "type": tipo,
            "mood": mood,
            "activity": activity,
            "match_strategy": None,
            "genre": genre,
            "confidence": "high" if tipo != "unknown" else "low"
        }

    # Ejecutar fallback
    result = smart_fallback()
    
    # Si el fallback tiene alta confianza O si OpenAI no está disponible, devolver directo
    if result.get("confidence") == "high" or not settings.openai_api_key:
        logger.info(f"✅ Fallback: type={result['type']}, mood={result['mood']}, genre={result['genre']}")
        return result

    # ===== USAR OPENAI SOLO SI ES NECESARIO =====
    
    # Control de rate limit
    now = time.time()
    
    # Reset contador cada minuto
    if now - _OPENAI_MINUTE_START > 60:
        _OPENAI_CALLS_THIS_MINUTE = 0
        _OPENAI_MINUTE_START = now
    
    # Si ya hicimos 2 llamadas este minuto, no llamar más (dejar margen)
    if _OPENAI_CALLS_THIS_MINUTE >= 2:
        logger.warning("⚠️ Rate limit preventivo - usando fallback")
        return result
    
    # Esperar al menos 2 segundos entre llamadas
    if now - _LAST_OPENAI_CALL < 2.0:
        time.sleep(2.0 - (now - _LAST_OPENAI_CALL))

    try:
        client = OpenAI(api_key=settings.openai_api_key)

        system_prompt = """Analizá el mensaje y devolvé SOLO JSON:
{
  "type": "movie" | "series" | "music" | "unknown",
  "mood": string | null,
  "activity": string | null,
  "genre": string | null
}

Sé conciso. Solo detectá lo explícito."""

        logger.info(f"🤖 OpenAI - parse: '{text[:40]}...'")

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=100
        )

        _LAST_OPENAI_CALL = time.time()
        _OPENAI_CALLS_THIS_MINUTE += 1
        
        content = completion.choices[0].message.content.strip()
        parsed = json.loads(content)

        return {
            "type": parsed.get("type") or result["type"],
            "mood": parsed.get("mood") or result["mood"],
            "activity": parsed.get("activity") or result["activity"],
            "match_strategy": None,
            "genre": parsed.get("genre") or result["genre"],
        }

    except Exception as e:
        logger.error(f"❌ Error OpenAI: {e}")
        return result


def detect_strategy_from_text(text: str) -> str | None:
    """Detecta estrategia del texto."""
    t = text.lower()
    if any(w in t for w in ["acompañe", "acompañar", "igual", "match", "siga", "mantenga", "acorde"]):
        return "match"
    if any(w in t for w in ["cambie", "cambiar", "contrario", "contrast", "levant", "anim", "diferente"]):
        return "contrast"
    return None


def infer_strategy_with_openai(reply_text: str, previous_parsed: Dict[str, Any]) -> str | None:
    """
    Infiere estrategia (con fallback simple sin OpenAI).
    """
    # Primero intentar con palabras clave
    strategy = detect_strategy_from_text(reply_text)
    if strategy:
        return strategy
    
    # Si no hay OpenAI, devolver None
    if not settings.openai_api_key:
        return None

    # NO USAR OPENAI AQUÍ para evitar rate limits
    # El usuario tendrá que ser más explícito
    return None


# =========================
# SPOTIFY
# =========================

SPOTIFY_TOKEN: str | None = None
SPOTIFY_TOKEN_EXPIRES_AT: float = 0.0


def get_spotify_token() -> str:
    """Obtiene token de Spotify (con cache)."""
    global SPOTIFY_TOKEN, SPOTIFY_TOKEN_EXPIRES_AT

    if SPOTIFY_TOKEN and time.time() < SPOTIFY_TOKEN_EXPIRES_AT - 60:
        return SPOTIFY_TOKEN

    if not settings.spotify_client_id or not settings.spotify_client_secret:
        raise RuntimeError("SPOTIFY_CLIENT_ID/SECRET no configurados")

    logger.info("🎧 Spotify - nuevo token")

    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(settings.spotify_client_id, settings.spotify_client_secret),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    SPOTIFY_TOKEN = data["access_token"]
    SPOTIFY_TOKEN_EXPIRES_AT = time.time() + data.get("expires_in", 3600)
    return SPOTIFY_TOKEN


# =========================
# TMDB
# =========================

TMDB_BASE_URL = "https://api.themoviedb.org/3"

MOVIE_GENRES = {
    "accion": 28, "acción": 28, "aventura": 12, "animacion": 16, "animación": 16,
    "comedia": 35, "crimen": 80, "documental": 99, "drama": 18, "familia": 10751,
    "fantasia": 14, "fantasía": 14, "terror": 27, "horror": 27, "misterio": 9648,
    "romance": 10749, "ciencia ficcion": 878, "ciencia ficción": 878, "thriller": 53,
}

TV_GENRES = {
    "comedia": 35, "animacion": 16, "animación": 16, "drama": 18,
    "misterio": 9648, "crimen": 80, "familia": 10751,
    "ciencia ficcion": 10765, "ciencia ficción": 10765,
}


def _resolve_movie_genre_id(name: str | None) -> int | None:
    return MOVIE_GENRES.get(name.lower().strip()) if name else None


def _resolve_tv_genre_id(name: str | None) -> int | None:
    return TV_GENRES.get(name.lower().strip()) if name else None


def _tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not settings.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY no configurada")

    url = f"{TMDB_BASE_URL}{path}"
    resp = requests.get(
        url,
        params={"api_key": settings.tmdb_api_key, **params},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def _tmdb_get_providers(kind: str, tmdb_id: int, region: str = "AR") -> str:
    try:
        data = _tmdb_get(f"/{kind}/{tmdb_id}/watch/providers", {})
        country = data.get("results", {}).get(region.upper())
        if not country:
            return "No disponible"

        names = []
        for key in ("flatrate", "ads", "rent", "buy"):
            for p in country.get(key, []):
                name = p.get("provider_name")
                if name and name not in names:
                    names.append(name)

        return ", ".join(names[:3]) if names else "No disponible"
    except:
        return "No disponible"


def get_movie_recommendations(parsed: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    if not settings.tmdb_api_key:
        return []

    region = settings.region.upper()
    genre_name = parsed.get("genre")
    mood = (parsed.get("mood") or "").lower()
    strategy = parsed.get("match_strategy")

    genre_id = _resolve_movie_genre_id(genre_name)

    if not genre_id:
        if "triste" in mood and strategy == "contrast":
            genre_id = MOVIE_GENRES["comedia"]
        elif "estres" in mood and strategy == "contrast":
            genre_id = MOVIE_GENRES["comedia"]

    try:
        params = {
            "language": "es-ES",
            "region": region,
            "sort_by": "popularity.desc",
            "include_adult": "false",
        }
        if genre_id:
            params["with_genres"] = genre_id

        data = _tmdb_get("/discover/movie", params)
        results = data.get("results", [])[:limit * 2]
        recs = []

        for r in results:
            details = _tmdb_get(f"/movie/{r['id']}", {"language": "es-ES"})
            
            recs.append({
                "title": details.get("title", "Sin título"),
                "overview": details.get("overview", "Sin sinopsis"),
                "genre": ", ".join(g["name"] for g in details.get("genres", [])[:2]) or "N/D",
                "duration": f"{details.get('runtime', 'N/D')} min",
                "year": details.get("release_date", "")[:4] or "N/D",
                "platforms": _tmdb_get_providers("movie", r["id"], region),
            })

            if len(recs) >= limit:
                break

        return recs
    except Exception as e:
        logger.error(f"❌ Error TMDB movies: {e}")
        return []


def get_series_recommendations(parsed: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    if not settings.tmdb_api_key:
        return []

    region = settings.region.upper()
    genre_name = parsed.get("genre")
    mood = (parsed.get("mood") or "").lower()
    strategy = parsed.get("match_strategy")

    genre_id = _resolve_tv_genre_id(genre_name)

    if not genre_id and "triste" in mood and strategy == "contrast":
        genre_id = TV_GENRES.get("comedia")

    try:
        params = {
            "language": "es-ES",
            "region": region,
            "sort_by": "popularity.desc",
            "include_adult": "false",
        }
        if genre_id:
            params["with_genres"] = genre_id

        data = _tmdb_get("/discover/tv", params)
        results = data.get("results", [])[:limit * 2]
        recs = []

        for r in results:
            details = _tmdb_get(f"/tv/{r['id']}", {"language": "es-ES"})
            
            runtime = details.get("episode_run_time", [])
            duration = f"{runtime[0]} min/ep" if runtime else "N/D"

            recs.append({
                "title": details.get("name", "Sin título"),
                "overview": details.get("overview", "Sin sinopsis"),
                "genre": ", ".join(g["name"] for g in details.get("genres", [])[:2]) or "N/D",
                "seasons": details.get("number_of_seasons", "N/D"),
                "episodes": details.get("number_of_episodes", "N/D"),
                "duration": duration,
                "year": details.get("first_air_date", "")[:4] or "N/D",
                "platforms": _tmdb_get_providers("tv", r["id"], region),
            })

            if len(recs) >= limit:
                break

        return recs
    except Exception as e:
        logger.error(f"❌ Error TMDB series: {e}")
        return []


def get_music_recommendations(parsed: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    try:
        token = get_spotify_token()
    except Exception as e:
        logger.error(f"❌ Spotify token: {e}")
        return []

    mood = (parsed.get("mood") or "").lower()
    activity = (parsed.get("activity") or "").lower()
    strategy = parsed.get("match_strategy")
    genre_name = (parsed.get("genre") or "").lower()

    vibe = ""
    if "correr" in activity or "gym" in activity:
        vibe = "workout energy"
    elif "estudiar" in activity or "trabajar" in activity:
        vibe = "focus study"
    elif "cocinar" in activity:
        vibe = "cooking chill"
    elif "dormir" in activity:
        vibe = "sleep"
    elif "triste" in mood and strategy == "contrast":
        vibe = "happy upbeat"
    elif "estres" in mood and strategy == "contrast":
        vibe = "chill relax"
    elif "relajad" in mood:
        vibe = "chill"
    else:
        vibe = mood or ""

    query_parts = [p for p in [genre_name, vibe] if p]
    query = " ".join(query_parts) or "popular"

    logger.info(f"🎧 Spotify query: '{query}'")

    try:
        resp = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": query, "type": "track", "limit": limit, "market": "AR"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("tracks", {}).get("items", [])

        recs = []
        for item in items:
            artists = item.get("artists", [])
            artist = artists[0] if artists else None
            
            # Simplificar: no buscar géneros de artista para evitar rate limits
            genres = "Música"

            recs.append({
                "title": item["name"],
                "artist": artist["name"] if artist else "Desconocido",
                "genres": genres,
                "url": item.get("external_urls", {}).get("spotify", ""),
            })

        return recs
    except Exception as e:
        logger.error(f"❌ Spotify search: {e}")
        return []


def save_conversation_history(user_id: str, user_text: str, bot_text: str, parsed: Dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = []
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            pass

    data.append({
        "user_id": user_id,
        "user_message": user_text,
        "bot_response": bot_text,
        "parsed": parsed,
        "timestamp": time.time(),
    })

    if len(data) > 100:
        data = data[-100:]

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)