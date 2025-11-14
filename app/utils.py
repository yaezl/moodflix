# app/utils.py
from typing import Dict, Any, List
import json
from pathlib import Path
import requests
from openai import OpenAI
from .config import settings

HISTORY_PATH = Path("data/conversation_history.json")


def parse_user_intent_with_openai(text: str) -> Dict[str, Any]:
    """
    Usa OpenAI (si hay API key) para interpretar:
      - type: music | movie | series
      - mood: estado de ánimo o vibe
      - activity: actividad (correr, estudiar, etc.) si aplica
      - match_strategy: match | contrast | null (si el usuario ya lo dijo)
    Si no hay API key o algo falla, usa una heurística básica.
    """

    # --- Fallback heurístico (versión vieja) por si no hay clave ---
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
        elif "relaj" in text_lower or "dormir" in text_lower:
            mood = "relajado"
            actividad = "descanso"
        else:
            mood = "neutral"
            actividad = None

        return {
            "type": tipo,
            "mood": mood,
            "activity": actividad,
            "match_strategy": None,
        }

    # Si no hay API key, uso fallback
    if not settings.openai_api_key:
        return fallback()

    try:
        client = OpenAI(api_key=settings.openai_api_key)

        system_prompt = (
    "Sos un analizador de intención para un bot de recomendaciones de entretenimiento.\n"
    "Dado un mensaje en español del usuario, devolvés SOLO un JSON con:\n"
    "{\n"
    '  \"type\": \"music\" | \"movie\" | \"series\",\n'
    '  \"mood\": \"texto corto en español (ej: energético, triste, relajado, ansioso, enojado, cansado, neutral)\",\n'
    '  \"activity\": \"actividad principal si se menciona (ej: correr, estudiar, limpiar) o null\",\n'
    '  \"match_strategy\": \"match\" | \"contrast\" | null\n'
    "}\n\n"
    "- \"match_strategy\" = \"match\" si el usuario quiere algo que siga el mismo mood o lo refuerce "
    "(ej: \"seguir tranqui\", \"algo triste para llorar\", \"algo relajado\").\n"
    "- \"match_strategy\" = \"contrast\" si el usuario quiere cambiar de estado de ánimo o levantarlo "
    "(ej: \"para animarme\", \"algo para levantarme\", \"algo que me saque de este bajón\").\n"
    "- Si no se entiende claramente, usar null.\n"
    "No agregues nada fuera del JSON."
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

        # Intentamos parsear el JSON que devuelve el modelo
        parsed = json.loads(content)

        # Normalizamos un poco por si el modelo devuelve cosas raras
        tipo = parsed.get("type", "music")
        if tipo not in ["music", "movie", "series"]:
            tipo = "music"

        mood = parsed.get("mood") or "neutral"
        activity = parsed.get("activity")
        match_strategy = parsed.get("match_strategy")
        if match_strategy not in ["match", "contrast"]:
            match_strategy = None

        return {
            "type": tipo,
            "mood": mood,
            "activity": activity,
            "match_strategy": None,
        }

    except Exception as e:
        # Si algo sale mal (error de red, JSON inválido, etc.), no rompemos el bot
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
    Devuelve hasta `limit` películas desde TMDB, en base a mood/actividad.
    Cada item tiene: title, year, genre, duration, overview, platforms.
    """
    if not settings.tmdb_api_key:
        print("TMDB_API_KEY no configurada, sin recomendaciones de películas.")
        return []

    region = (settings.region or "AR").upper()

    # Armamos un query simple usando actividad + mood
    query_parts: list[str] = []
    activity = parsed.get("activity")
    mood = parsed.get("mood")

    if activity:
        query_parts.append(activity)
    if mood and mood not in ["neutral", "neutro"]:
        query_parts.append(mood)

    query = " ".join(query_parts) or "popular"

    try:
        # 1) Buscar películas
        search_data = _tmdb_get(
            "/search/movie",
            {
                "query": query,
                "language": "es-ES",
                "region": region,
                "include_adult": "false",
            },
        )
        results = search_data.get("results", [])[: limit * 2]  # agarramos un poquito más por si alguna falla

        recs: List[Dict[str, Any]] = []

        for r in results:
            movie_id = r["id"]

            # 2) Detalles: duración, géneros, sinopsis, año
            details = _tmdb_get(
                f"/movie/{movie_id}",
                {"language": "es-ES"},
            )

            title = details.get("title") or r.get("title") or "Sin título"
            overview = details.get("overview") or r.get("overview") or "Sin sinopsis disponible."
            genres = ", ".join(g.get("name") for g in details.get("genres", [])) or "N/D"
            runtime = details.get("runtime")
            duration = f"{runtime} min" if runtime else "N/D"
            release_date = details.get("release_date") or ""
            year = release_date[:4] if release_date else "N/D"

            # 3) Plataformas (AR)
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
    Cada item tiene: title, year, genre, seasons, episodes, duration, overview, platforms.
    """
    if not settings.tmdb_api_key:
        print("TMDB_API_KEY no configurada, sin recomendaciones de series.")
        return []

    region = (settings.region or "AR").upper()

    query_parts: list[str] = []
    activity = parsed.get("activity")
    mood = parsed.get("mood")

    if activity:
        query_parts.append(activity)
    if mood and mood not in ["neutral", "neutro"]:
        query_parts.append(mood)

    query = " ".join(query_parts) or "popular"

    try:
        # 1) Buscar series
        search_data = _tmdb_get(
            "/search/tv",
            {
                "query": query,
                "language": "es-ES",
                "region": region,
                "include_adult": "false",
            },
        )
        results = search_data.get("results", [])[: limit * 2]

        recs: List[Dict[str, Any]] = []

        for r in results:
            series_id = r["id"]

            # 2) Detalles: temporadas, episodios, duración promedio
            details = _tmdb_get(
                f"/tv/{series_id}",
                {"language": "es-ES"},
            )

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

            # 3) Plataformas (AR)
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
