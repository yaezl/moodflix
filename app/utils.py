# app/utils.py
from typing import Dict, Any, List
import json
from pathlib import Path
import requests
import time
from openai import OpenAI
from .config import settings

HISTORY_PATH = Path("data/conversation_history.json")


def parse_user_intent_with_openai(text: str) -> Dict[str, Any]:
    """
    Usa OpenAI (si hay API key) para interpretar:
      - type: music | movie | series
      - mood: estado de ánimo (triste, estresado, relajado, feliz, etc.)
      - activity: actividad (correr, estudiar, dormir...) o null
      - match_strategy: match | contrast | null  (IGNORAMOS esto en el primer mensaje)
      - genre: si pide explícitamente un género (terror, comedia, drama, romance, acción, etc.)
    Si no hay API key o algo falla, usa una heurística básica.
    """

    # --- Fallback heurístico por si no hay clave o falla OpenAI ---
    def fallback():
        text_lower = text.lower()
        tipo = "music"
        if "peli" in text_lower or "película" in text_lower or "pelicula" in text_lower:
            tipo = "movie"
        elif "serie" in text_lower or "capitulo" in text_lower or "capítulo" in text_lower:
            tipo = "series"

        if "correr" in text_lower or "gym" in text_lower or "entrenar" in text_lower:
            mood = "energético"
            actividad = "ejercicio"
        elif "triste" in text_lower or "bajon" in text_lower or "bajón" in text_lower:
            mood = "triste"
            actividad = None
        elif "estres" in text_lower or "estrés" in text_lower or "ansiosa" in text_lower or "ansioso" in text_lower:
            mood = "estresado/ansioso"
            actividad = None
        elif "relaj" in text_lower or "dormir" in text_lower:
            mood = "relajado"
            actividad = "descanso"
        else:
            mood = "neutral"
            actividad = None

        # género básico por palabras
        genre = None
        for g in ["terror", "miedo", "comedia", "drama", "romance", "accion", "acción",
          "ciencia ficción", "ciencia ficcion", "pop", "rock", "reggaeton", "reguetón", "jazz"]:
           if g in text_lower:
               genre = g
               break


        return {
            "type": tipo,
            "mood": mood,
            "activity": actividad,
            "match_strategy": None,
            "genre": genre,
        }

    if not settings.openai_api_key:
        return fallback()

    try:
        client = OpenAI(api_key=settings.openai_api_key)

        system_prompt = (
            "Sos un analizador de intención para un bot de recomendaciones de entretenimiento "
            "(música, películas y series).\n"
            "Debés devolver SOLO un JSON válido con esta forma:\n\n"
            "{\n"
            '  \"type\": \"music\" | \"movie\" | \"series\",\n'
            '  \"mood\": \"texto corto en español (ej: triste, feliz, relajado, estresado, ansioso, enojado, cansado, neutral)\",\n'
            '  \"activity\": \"actividad principal si se menciona (ej: correr, estudiar, limpiar, dormir) o null\",\n'
            '  \"match_strategy\": \"match\" | \"contrast\" | null,\n'
            '  \"genre\": \"si el usuario pide explícitamente un género (terror, comedia, drama, romance, acción, ciencia ficción, animación, documental, familiar), sino null\"\n'
            "}\n\n"
            "- Si el mensaje habla principalmente de canciones, playlists, temas → type = \"music\".\n"
            "- Si habla de película, cine, peli → type = \"movie\".\n"
            "- Si habla de serie, temporadas, capítulos, maratonear → type = \"series\".\n"
            "- Detectá el estado de ánimo REAL, por ejemplo:\n"
            "    * \"estresado\", \"estrés\", \"estres\", \"ansiosa\", \"ansioso\" → mood = \"estresado/ansioso\" (NUNCA \"relajado\").\n"
            "    * \"triste\", \"bajoneada\", \"bajón\" → mood = \"triste\".\n"
            "    * \"relajado\", \"tranqui\", \"relajada\" → mood = \"relajado\".\n"
            "    * Si no se menciona claramente un estado de ánimo → \"neutral\".\n"
            "- Si el usuario pide un género explícito (ej: \"películas de terror\", \"series de comedia\"), completá \"genre\" con ese género estandarizado "
            "en minúsculas (terror, comedia, drama, romance, acción, ciencia ficción, animación, documental, familiar).\n"
            "- Si el mensaje es principalmente pedir un género y casi no habla de emociones, podés dejar mood = \"neutral\".\n"
            "- \"match_strategy\":\n"
            "    * \"match\" si el usuario ya dijo que quiere algo acorde a su estado.\n"
            "    * \"contrast\" si dijo claramente que quiere cambiar el estado (\"algo para levantarme\", \"para animarme\", \"algo diferente\").\n"
            "    * null si no está claro.\n"
            "No agregues ningún texto fuera del JSON."
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )

        content = completion.choices[0].message.content.strip()
        parsed = json.loads(content)

        tipo = parsed.get("type", "music")
        if tipo not in ["music", "movie", "series"]:
            tipo = "music"

        mood = parsed.get("mood") or "neutral"
        activity = parsed.get("activity")
        genre = parsed.get("genre")

        # Ignoramos match_strategy en el primer análisis: siempre preguntamos o lo inferimos después
        return {
            "type": tipo,
            "mood": mood,
            "activity": activity,
            "match_strategy": None,
            "genre": genre,
        }

    except Exception as e:
        print("Error en parse_user_intent_with_openai:", e)
        return fallback()

def detect_strategy_from_text(text: str) -> str | None:
    """
    Intenta ver si el usuario ya dijo que quiere acompañar (match)
    o cambiar el ánimo (contrast) en este mismo mensaje.
    """
    t = text.lower()
    # Acompañar / mantener estado
    if "acompañe" in t or "acompañar" in t or "igual" in t or "match" in t:
        return "match"
    # Cambiar estado / contrario
    if "cambie" in t or "cambiar" in t or "contrario" in t or "contraste" in t:
        return "contrast"
    return None

def infer_strategy_with_openai(reply_text: str, previous_parsed: Dict[str, Any]) -> str | None:
    """
    Usa OpenAI para interpretar si la respuesta del usuario a la pregunta
    '¿acompañar o cambiar?' implica match o contrast.
    Si no hay API key o hay error, devuelve None.
    """
    if not settings.openai_api_key:
        return None

    try:
        client = OpenAI(api_key=settings.openai_api_key)

        system_prompt = (
            "Sos un clasificador. Te doy el pedido original del usuario y su respuesta a la pregunta "
            "\"¿querés que las recomendaciones acompañen tu estado de ánimo o lo cambien?\".\n"
            "Devolvés SOLO un JSON con la forma:\n"
            "{ \"match_strategy\": \"match\" | \"contrast\" }\n\n"
            "- \"match\" si la respuesta indica que quiere mantener o acompañar el mood.\n"
            "- \"contrast\" si la respuesta indica que quiere cambiar el mood, levantarlo, algo diferente.\n"
            "No agregues nada fuera del JSON."
        )

        original_summary = (
            f"Tipo: {previous_parsed.get('type')}, "
            f"mood: {previous_parsed.get('mood')}, "
            f"actividad: {previous_parsed.get('activity')}"
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pedido original: {original_summary}"},
                {"role": "user", "content": f"Respuesta del usuario: {reply_text}"},
            ],
            temperature=0,
        )

        content = completion.choices[0].message.content.strip()
        parsed = json.loads(content)
        strategy = parsed.get("match_strategy")
        if strategy in ["match", "contrast"]:
            return strategy
        return None

    except Exception as e:
        print("Error en infer_strategy_with_openai:", e)
        return None

# =========================
# TMDB – Helpers y películas/series
# =========================

TMDB_BASE_URL = "https://api.themoviedb.org/3"

# =========================
# SPOTIFY – Auth y helpers
# =========================

SPOTIFY_TOKEN: str | None = None
SPOTIFY_TOKEN_EXPIRES_AT: float = 0.0


def get_spotify_token() -> str:
    """
    Usa Client Credentials para obtener un access token de Spotify.
    Cachea el token en memoria hasta que expire.
    """
    global SPOTIFY_TOKEN, SPOTIFY_TOKEN_EXPIRES_AT

    if SPOTIFY_TOKEN and time.time() < SPOTIFY_TOKEN_EXPIRES_AT - 60:
        return SPOTIFY_TOKEN

    if not settings.spotify_client_id or not settings.spotify_client_secret:
        raise RuntimeError("SPOTIFY_CLIENT_ID o SPOTIFY_CLIENT_SECRET no configurados en .env")

    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(settings.spotify_client_id, settings.spotify_client_secret),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    SPOTIFY_TOKEN = data["access_token"]
    expires_in = data.get("expires_in", 3600)
    SPOTIFY_TOKEN_EXPIRES_AT = time.time() + expires_in
    return SPOTIFY_TOKEN


# =========================
# TMDB – Mapas de géneros
# =========================

MOVIE_GENRES = {
    "accion": 28,
    "acción": 28,
    "aventura": 12,
    "animacion": 16,
    "animación": 16,
    "comedia": 35,
    "crimen": 80,
    "documental": 99,
    "drama": 18,
    "familia": 10751,
    "fantasia": 14,
    "fantasía": 14,
    "historia": 36,
    "terror": 27,
    "horror": 27,
    "misterio": 9648,
    "musica": 10402,
    "música": 10402,
    "romance": 10749,
    "ciencia ficcion": 878,
    "ciencia ficción": 878,
    "sci-fi": 878,
    "suspenso": 53,
    "thriller": 53,
}

TV_GENRES = {
    "comedia": 35,
    "animacion": 16,
    "animación": 16,
    "drama": 18,
    "misterio": 9648,
    "crimen": 80,
    "familia": 10751,
    "scifi": 10765,
    "ciencia ficcion": 10765,
    "ciencia ficción": 10765,
}


def _resolve_movie_genre_id(name: str | None) -> int | None:
    if not name:
        return None
    n = name.lower().strip()
    return MOVIE_GENRES.get(n)


def _resolve_tv_genre_id(name: str | None) -> int | None:
    if not name:
        return None
    n = name.lower().strip()
    return TV_GENRES.get(n)


def _tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper para llamar a TMDB con api_key y manejar errores básicos.
    """
    if not settings.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY no configurada en .env")

    url = f"{TMDB_BASE_URL}{path}"
    all_params = {
        "api_key": settings.tmdb_api_key,
        **params,
    }
    resp = requests.get(url, params=all_params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _tmdb_get_providers(kind: str, tmdb_id: int, region: str = "AR") -> str:
    """
    Obtiene proveedores (plataformas) para una película o serie en una región.
    kind: "movie" o "tv"
    """
    try:
        data = _tmdb_get(f"/{kind}/{tmdb_id}/watch/providers", {})
        results = data.get("results", {})
        country = results.get(region.upper())
        if not country:
            return "No se encontraron plataformas para tu región"

        names: list[str] = []
        for key in ("flatrate", "ads", "rent", "buy"):
            for p in country.get(key, []) or []:
                name = p.get("provider_name")
                if name and name not in names:
                    names.append(name)

        return ", ".join(names) if names else "No se encontraron plataformas para tu región"
    except Exception as e:
        print("Error obteniendo providers TMDB:", e)
        return "No se pudieron obtener las plataformas"


def get_movie_recommendations(parsed: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """
    Devuelve hasta `limit` películas desde TMDB.
    Usa primero el género (si viene de la intención), y si no hay género,
    decide géneros según mood + estrategia (match/contrast).
    """
    if not settings.tmdb_api_key:
        print("TMDB_API_KEY no configurada, sin recomendaciones de películas.")
        return []

    region = (settings.region or "AR").upper()
    genre_name = parsed.get("genre")
    mood = (parsed.get("mood") or "").lower()
    strategy = parsed.get("match_strategy")

    genre_id = _resolve_movie_genre_id(genre_name)

    # Si no hay género explícito, elegimos alguno según mood + estrategia
    if genre_id is None:
        if "triste" in mood and strategy == "contrast":
            # está triste y quiere cambiar → comedia/romance
            genre_id = MOVIE_GENRES["comedia"]
        elif "estres" in mood or "ansioso" in mood or "ansiosa" in mood:
            # estresada: si quiere cambiar → comedia
            if strategy == "contrast":
                genre_id = MOVIE_GENRES["comedia"]
            else:
                genre_id = MOVIE_GENRES.get("thriller") or MOVIE_GENRES["drama"]
        elif "relajado" in mood or "tranqui" in mood:
            genre_id = MOVIE_GENRES["drama"]
        else:
            # default: algo popular sin género particular
            genre_id = None

    try:
        if genre_id:
            params = {
                "language": "es-ES",
                "region": region,
                "sort_by": "popularity.desc",
                "with_genres": genre_id,
                "include_adult": "false",
            }
            data = _tmdb_get("/discover/movie", params)
        else:
            # sin género definido: usamos discover por popularidad general
            params = {
                "language": "es-ES",
                "region": region,
                "sort_by": "popularity.desc",
                "include_adult": "false",
            }
            data = _tmdb_get("/discover/movie", params)

        results = data.get("results", [])[: limit * 2]
        recs: List[Dict[str, Any]] = []

        for r in results:
            movie_id = r["id"]

            details = _tmdb_get(f"/movie/{movie_id}", {"language": "es-ES"})

            title = details.get("title") or r.get("title") or "Sin título"
            overview = details.get("overview") or r.get("overview") or "Sin sinopsis disponible."
            genres = ", ".join(g.get("name") for g in details.get("genres", [])) or "N/D"
            runtime = details.get("runtime")
            duration = f"{runtime} min" if runtime else "N/D"
            release_date = details.get("release_date") or ""
            year = release_date[:4] if release_date else "N/D"
            platforms = _tmdb_get_providers("movie", movie_id, region=region)

            recs.append(
                {
                    "title": title,
                    "overview": overview,
                    "genre": genres,
                    "duration": duration,
                    "year": year,
                    "platforms": platforms,
                }
            )

            if len(recs) >= limit:
                break

        return recs

    except Exception as e:
        print("Error en get_movie_recommendations:", e)
        return []


def get_series_recommendations(parsed: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """
    Devuelve hasta `limit` series desde TMDB.
    Usa primero género explícito, si no hay usa mood + estrategia para elegir género aproximado.
    """
    if not settings.tmdb_api_key:
        print("TMDB_API_KEY no configurada, sin recomendaciones de series.")
        return []

    region = (settings.region or "AR").upper()
    genre_name = parsed.get("genre")
    mood = (parsed.get("mood") or "").lower()
    strategy = parsed.get("match_strategy")

    genre_id = _resolve_tv_genre_id(genre_name)

    if genre_id is None:
        if "triste" in mood and strategy == "contrast":
            genre_id = TV_GENRES.get("comedia")
        elif "relajado" in mood or "tranqui" in mood:
            genre_id = TV_GENRES.get("drama")
        else:
            genre_id = None

    try:
        if genre_id:
            params = {
                "language": "es-ES",
                "region": region,
                "sort_by": "popularity.desc",
                "with_genres": genre_id,
                "include_adult": "false",
            }
            data = _tmdb_get("/discover/tv", params)
        else:
            params = {
                "language": "es-ES",
                "region": region,
                "sort_by": "popularity.desc",
                "include_adult": "false",
            }
            data = _tmdb_get("/discover/tv", params)

        results = data.get("results", [])[: limit * 2]
        recs: List[Dict[str, Any]] = []

        for r in results:
            series_id = r["id"]

            details = _tmdb_get(f"/tv/{series_id}", {"language": "es-ES"})

            title = details.get("name") or r.get("name") or "Sin título"
            overview = details.get("overview") or r.get("overview") or "Sin sinopsis disponible."
            genres = ", ".join(g.get("name") for g in details.get("genres", [])) or "N/D"
            seasons = details.get("number_of_seasons") or "N/D"
            episodes = details.get("number_of_episodes") or "N/D"
            runtime_list = details.get("episode_run_time") or []
            ep_runtime = runtime_list[0] if runtime_list else None
            duration = f"{ep_runtime} min por episodio" if ep_runtime else "N/D"
            first_air_date = details.get("first_air_date") or ""
            year = first_air_date[:4] if first_air_date else "N/D"
            platforms = _tmdb_get_providers("tv", series_id, region=region)

            recs.append(
                {
                    "title": title,
                    "overview": overview,
                    "genre": genres,
                    "seasons": seasons,
                    "episodes": episodes,
                    "duration": duration,
                    "year": year,
                    "platforms": platforms,
                }
            )

            if len(recs) >= limit:
                break

        return recs

    except Exception as e:
        print("Error en get_series_recommendations:", e)
        return []

def get_music_recommendations(parsed: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """
    Devuelve hasta `limit` canciones de Spotify en base a:
      - mood (triste, relajado, estresado, etc.)
      - activity (correr, estudiar, dormir)
      - genre (si el usuario pidió un género concreto)
      - strategy (match / contrast) → para elegir vibes opuestas o similares.

    Cada item tiene:
      - title
      - artist
      - genres (string)
      - url (link a Spotify)
    """
    try:
        token = get_spotify_token()
    except Exception as e:
        print("Error obteniendo token de Spotify:", e)
        return []

    mood = (parsed.get("mood") or "").lower()
    activity = (parsed.get("activity") or "").lower()
    strategy = parsed.get("match_strategy")
    genre_name = (parsed.get("genre") or "").lower()

    # 1) Definimos una "vibe" interna según mood + actividad + estrategia
    vibe = ""

    # Actividades
    if "correr" in activity or "gym" in activity or "entrenar" in activity:
        vibe = "workout upbeat"
    elif "estudiar" in activity or "trabajar" in activity:
        vibe = "focus study"
    elif "dormir" in activity or "relajar" in activity:
        vibe = "sleep chill"

    # Mood + estrategia
    if not vibe:
        if "triste" in mood and strategy == "contrast":
            vibe = "happy upbeat"
        elif ("estres" in mood or "ansioso" in mood or "ansiosa" in mood) and strategy == "contrast":
            vibe = "chill relax"
        elif "relajado" in mood or "tranqui" in mood:
            vibe = "chill"
        elif "enojado" in mood and strategy == "contrast":
            vibe = "calm"
        else:
            vibe = mood or "mood"

    # 2) Armamos el query de Spotify
    query_parts: List[str] = []

    # Género que pidió el usuario (pop, rock, etc.)
    if genre_name:
        query_parts.append(genre_name)

    # Vibe interno (no ponemos la palabra "correr", sino "workout upbeat", etc.)
    if vibe:
        query_parts.append(vibe)

    query = " ".join(query_parts).strip() or "popular"

    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = requests.get(
            "https://api.spotify.com/v1/search",
            headers=headers,
            params={
                "q": query,
                "type": "track",
                "limit": limit,
                "market": "AR",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("tracks", {}).get("items", [])
        recommendations: List[Dict[str, Any]] = []

        for item in items:
            track_name = item["name"]
            artists = item.get("artists", [])
            first_artist = artists[0] if artists else None
            artist_name = first_artist["name"] if first_artist else "Artista desconocido"
            track_url = item.get("external_urls", {}).get("spotify", "")

            # Intentamos obtener géneros del artista
            artist_genres = "N/D"
            if first_artist and first_artist.get("id"):
                try:
                    artist_resp = requests.get(
                        f"https://api.spotify.com/v1/artists/{first_artist['id']}",
                        headers=headers,
                        timeout=10,
                    )
                    artist_resp.raise_for_status()
                    artist_data = artist_resp.json()
                    genres = artist_data.get("genres", [])
                    if genres:
                        if len(genres) > 2:
                            genres = genres[:2]
                        artist_genres = ", ".join(genres)
                except Exception as e:
                    print("Error obteniendo género de artista Spotify:", e)

            recommendations.append(
                {
                    "title": track_name,
                    "artist": artist_name,
                    "genres": artist_genres,
                    "url": track_url,
                }
            )

            if len(recommendations) >= limit:
                break

        return recommendations

    except Exception as e:
        print("Error en get_music_recommendations:", e)
        return []



def save_conversation_history(
    user_id: str,
    user_text: str,
    bot_text: str,
    parsed: Dict[str, Any]
) -> None:
    """
    Guarda un registro simple en data/conversation_history.json
    para después mostrar en el informe.
    """
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                data: List[Dict[str, Any]] = json.load(f)
        except Exception:
            data = []
    else:
        data = []

    data.append({
        "user_id": user_id,
        "user_message": user_text,
        "bot_response": bot_text,
        "parsed": parsed,
    })

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
