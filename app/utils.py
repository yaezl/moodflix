from typing import Dict, Any, List, Literal, Optional
from pathlib import Path
import json
import time
import logging
import re 

import requests

from .config import settings, groq_client, TMDB_BASE_URL, TMDB_LANG

logger = logging.getLogger("moodflix")

# ------------------------------
# Historial de conversación
# ------------------------------

HISTORY_PATH = Path("data/conversation_history.json")


def save_conversation_history(
    user_id: str,
    user_text: str,
    bot_text: str,
    parsed: Dict[str, Any]
) -> None:
    """
    Guarda en data/conversation_history.json el historial básico de la conversación.
    Lo dejamos igual que antes para que puedas analizar después.
    """
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    data: List[Dict[str, Any]] = []
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ No se pudo leer el historial previo: {e}")
            data = []

    data.append(
        {
            "user_id": user_id,
            "user_message": user_text,
            "bot_response": bot_text,
            "parsed": parsed,
            "timestamp": time.time(),
        }
    )

    # Limitar tamaño del historial
    if len(data) > 300:
        data = data[-300:]

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------------
# Groq – helpers
# ------------------------------

def groq_chat(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:

    logger.info("📡 Request a Groq → %s...", user_prompt[:80])

    resp = groq_client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=512,
    )

    content = resp.choices[0].message.content

    logger.info("📡 Respuesta de Groq ← %s...", content[:80])

    return content

def groq_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Igual que groq_chat, pero asumiendo que el modelo responde SOLO JSON.
    Si falla el parseo, devuelve {}.
    Además, limpia fences tipo ```json ... ``` que a veces agrega el modelo.
    """
    content = groq_chat(system_prompt, user_prompt, temperature=0.0)

    cleaned = content.strip()

    if cleaned.startswith("```"):

        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)  
        cleaned = re.sub(r"\s*```$", "", cleaned)   

    try:
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError:

        preview = cleaned.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        logger.warning("⚠️ No se pudo parsear JSON desde Groq. Respuesta (inicio): %s", preview)
        return {}

def extract_slots_from_text(
    user_text: str,
    last_question: Optional[str] = None,
    prev_slots: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Llama a Groq para interpretar la intención y los slots del usuario.
    Usa contexto de la última pregunta para entender respuestas cortas
    tipo 'pocas', 'largos', 'conocida', etc.
    """
    prev_slots = prev_slots or {}
    last_question = last_question or "ninguna (podés inferir por el mensaje)"
    prev_slots_json = json.dumps(prev_slots, ensure_ascii=False)

    system_prompt = """Sos un asistente que SOLO devuelve JSON con este formato:

{{
  "intent": "recommendation" | "answer" | "other",
  "slots": {{
    "tipo_contenido": "movie" | "tv" | "indiferente" | null,
    "generos": ["comedia", "terror", "drama", ...],
    "tono": "liviano" | "intenso" | "emocional" | "indiferente" | null,
    "novedad": "nuevo" | "clasico" | "indiferente" | null,
    "duracion_peli": "corta" | "larga" | "indiferente" | null,
    "temporadas": "pocas" | "varias" | "indiferente" | null,
    "episodios_totales": "pocos" | "muchos" | "indiferente" | null,
    "duracion_capitulo": "cortos" | "largos" | "indiferente" | null,
    "contexto": "solo" | "pareja" | "amigxs" | "familia" | null,
    "fama": "conocida" | "joyita" | "indiferente" | null,
    "restricciones": [],
    "personas_like": [],
    "personas_dislike": [],
    "tematicas": [],
    "cantidad_recs": 1
  }}
}}

Contexto de la conversación:
- Última pregunta: "{last_question}"
- Slots actuales: {prev_slots_json}

REGLAS:

1. Respuestas cortas a preguntas:
   - Pregunta: "¿Pocas temporadas o varias?" → usuario: "pocas" → "temporadas": "pocas"
   - Pregunta: "¿Pocos capítulos o muchos?" → usuario: "pocos" → "episodios_totales": "pocos"
   - Pregunta: "¿Capítulos cortos o largos?" → usuario: "largos" → "duracion_capitulo": "largos"
   - Pregunta: "¿Muy conocido o poco vista?" → usuario: "bastante conocido" → "fama": "conocida"
   - Pregunta: "¿Solo, pareja, amigos o familia?" → usuario: "amigxs" → "contexto": "amigxs"

2. Sinónimos aceptados:
   - "pocas", "una temporada", "corta" → "pocas"
   - "varias", "muchas", "larga" → "varias"
   - "cortos", "cortitos", "20 minutos" → "cortos"
   - "largos", "45 minutos", "una hora" → "largos"
   - "nuevo", "moderno", "reciente" → "nuevo"
   - "clásico", "viejo", "antiguo" → "clasico"
   - "conocida", "popular", "famosa" → "conocida"
   - "joyita", "poco conocida", "joya oculta" → "joyita"

3. Indiferencia:
   Cuando el usuario diga: "me da igual", "indiferente", "no sé", "cualquiera", "como quieras", 
   "lo que vos digas", "no tengo preferencia", "sin preferencia", "me es indistinto", etc.
   → Asigna el slot correspondiente a "indiferente"
   
   Ejemplos:
   - Pregunta: "¿Pocas temporadas o varias?" → usuario: "me da igual" → "temporadas": "indiferente"
   - Pregunta: "¿Preferís nuevo o clásico?" → usuario: "cualquiera" → "novedad": "indiferente"

4. Intent:
   - "recommendation": cuando pide que recomiende algo o cambia de tipo (peli/serie)
   - "answer": cuando responde una de tus preguntas
   - "other": cuando habla de algo no relacionado

5. IMPORTANTE - SOLO INCLUYE SLOTS QUE EL USUARIO MENCIONÓ:
   - NO completes automáticamente con "indiferente" los slots que no aparecen en el mensaje.
   - Si el usuario solo dice "terror", devuelve SOLO:
     {{
       "intent": "answer",
       "slots": {{
         "generos": ["terror"]
       }}
     }}
   - NO devuelvas "novedad", "duracion_peli", "fama", etc. si el usuario no las mencionó.

6. Restricciones (si el usuario las menciona):
   - "no animada", "sin animación" → "restricciones": ["no_animacion"]
   - "no terror", "sin miedo" → "restricciones": ["no_terror"]
   - "no gore", "no sangre" → "restricciones": ["no_gore"]
   - "no romance", "sin romance" → "restricciones": ["no_romance"]
   - "no sci-fi", "sin fantasía" → "restricciones": ["no_scifi"]
   - "no crimen", "no policiales" → "restricciones": ["no_crimen"]
   - "no guerra", "no bélicas" → "restricciones": ["no_guerra"]

7. Contexto social (si el usuario lo menciona):
   - "solo", "sola", "solito" → "contexto": "solo"
   - "pareja", "novio", "novia" → "contexto": "pareja"
   - "amigxs", "amigos", "mis amigas" → "contexto": "amigxs"
   - "familia", "familiar" → "contexto": "familia"

8. Temáticas (si el usuario las menciona):
   - "sobrenatural" → "tematicas": ["sobrenatural"]
   - "vampiros" → "tematicas": ["vampiros"]
   - "hombres lobo" → "tematicas": ["hombres_lobo"]
   - "doctores", "médicos" → "tematicas": ["doctores"]
   - "abogados" → "tematicas": ["abogados"]
   - "guerra", "bélica" → "tematicas": ["guerra"]
   - "amistad", "amigos" → "tematicas": ["amigos"]
   - "carreras", "autos" → "tematicas": ["carreras_autos"]
   - "basada en hechos reales" → "tematicas": ["hechos_reales"]

Devolvé SIEMPRE solo el JSON, sin texto adicional ni ```.
""".format(last_question=last_question, prev_slots_json=prev_slots_json)

    user_prompt = f"Mensaje del usuario: {user_text}"

    data = groq_json(system_prompt, user_prompt)

    # Fallback seguro
    intent = data.get("intent", "other")
    slots = data.get("slots", {}) or {}

    # Aseguramos campos mínimos
    if "cantidad_recs" not in slots:
        slots["cantidad_recs"] = 1
    if "tematicas" not in slots:
        slots["tematicas"] = []
    if "restricciones" not in slots:
        slots["restricciones"] = []

    return {"intent": intent, "slots": slots}


def merge_slots(prev_slots: Dict[str, Any] | None,
                new_slots: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Fusiona los slots anteriores con los nuevos.

    Reglas:
    - None, "", [] => se ignoran siempre (no se guardan).
    - "indiferente":
        * si el slot estaba vacío -> se guarda "indiferente"
        * si ya había un valor concreto -> se mantiene el anterior.
    - cualquier otro valor pisa al anterior.
    
    IMPORTANTE: Solo procesa los slots que Groq devolvió.
    No completes nada automáticamente.
    """
    merged: Dict[str, Any] = dict(prev_slots or {})

    if not new_slots:
        return merged

    for key, new_val in new_slots.items():
        old_val = merged.get(key)

        # Ignorar valores realmente vacíos
        if new_val in (None, "", []):
            continue

        # Manejo especial de "indiferente"
        if new_val == "indiferente":
            # Solo guarda "indiferente" si no había nada antes
            if old_val in (None, "", [], "indiferente"):
                merged[key] = "indiferente"
            # si ya había algo concreto (no indiferente), mantener lo anterior
            continue

        # Para cualquier otro valor concreto, pisamos
        merged[key] = new_val

    return merged

# ------------------------------
# TMDB – géneros y requests
# ------------------------------

ContentType = Literal["movie", "tv"]

# Mapeo de géneros en español a IDs de TMDB (películas)
MOVIE_GENRES: Dict[str, int] = {
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
    "terror": 27,
    "horror": 27,
    "misterio": 9648,
    "romance": 10749,
    "ciencia ficcion": 878,
    "ciencia ficción": 878,
    "thriller": 53,
}

# Mapeo de géneros para series
TV_GENRES: Dict[str, int] = {
    "comedia": 35,
    "animacion": 16,
    "animación": 16,
    "drama": 18,
    "misterio": 9648,
    "crimen": 80,
    "familia": 10751,
    "ciencia ficcion": 10765,
    "ciencia ficción": 10765,
}

def _resolve_genre_ids(content_type: ContentType, generos: List[str]) -> List[int]:
    ids: List[int] = []
    genre_map = MOVIE_GENRES if content_type == "movie" else TV_GENRES

    for name in generos:
        if not name:
            continue
        key = name.lower().strip()
        gid = genre_map.get(key)
        if gid and gid not in ids:
            ids.append(gid)
    return ids

def _tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Request genérico a TMDB con logging.
    """
    if not settings.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY no configurada")

    url = f"{TMDB_BASE_URL}{path}"
    full_params = {"api_key": settings.tmdb_api_key, **params}

    logger.info("🎬 TMDB GET → %s | params=%s", url, full_params)

    resp = requests.get(url, params=full_params, timeout=10)
    resp.raise_for_status()

    data = resp.json()
    return data

def discover_tmdb(content_type: ContentType, slots: Dict[str, Any], page: int = 1) -> Dict[str, Any]:
    """
    Hace una búsqueda en TMDB usando los slots del usuario.
    No elige aún la mejor recomendación, solo trae resultados crudos.
    """

    restricciones = slots.get("restricciones") or []
    tematicas = slots.get("tematicas") or []

    # -----------------------------
    # Mapeos de restricciones → géneros TMDB
    # -----------------------------
    restriccion_a_generos = {
        "no_animacion": ["16"],
        "no_terror": ["27"],
        "no_romance": ["10749"],
        "no_scifi": ["878", "14"],
        "no_crimen": ["80"],
        "no_guerra": ["10752"],
    }

    # -----------------------------
    # Mapeos temáticos → búsqueda textual en TMDB
    # -----------------------------
    tema_a_query = {
        "sobrenatural": "supernatural",
        "vampiros": "vampire",
        "hombres_lobo": "werewolf",
        "doctores": "doctor hospital",
        "abogados": "lawyer courtroom",
        "guerra": "war soldier",
        "amigos": "friends friendship",
        "carreras_autos": "car racing",
        "hechos_reales": "based on true story",
    }

    # -----------------------------
    # Convertimos restricciones → géneros excluidos
    # -----------------------------
    without = []
    for r in restricciones:
        if r in restriccion_a_generos:
            without.extend(restriccion_a_generos[r])

    # -----------------------------
    # Construcción del diccionario base de params
    # -----------------------------
    params: Dict[str, Any] = {
        "language": TMDB_LANG,
        "region": settings.region.upper(),
        "include_adult": "false",
        "page": page,
    }

    # Aplicar exclusiones de géneros
    if without:
        params["without_genres"] = ",".join(without)

    # -----------------------------
    # Temáticas → búsqueda textual
    # -----------------------------
    query_text = ""
    if tematicas:
        partes = []
        for tema in tematicas:
            if tema in tema_a_query:
                partes.append(tema_a_query[tema])
        if partes:
            query_text = " ".join(partes)

    # -----------------------------
    # Filtros de géneros
    # -----------------------------
    generos = slots.get("generos") or []
    genre_ids = _resolve_genre_ids(content_type, generos)
    if genre_ids:
        params["with_genres"] = ",".join(str(g) for g in genre_ids)

    # -----------------------------
    # Filtro de novedad
    # -----------------------------
    novedad = slots.get("novedad")
    if content_type == "movie":
        if novedad == "nuevo":
            params["primary_release_date.gte"] = "2015-01-01"
        elif novedad == "clasico":
            params["primary_release_date.lte"] = "2005-12-31"
    else:
        if novedad == "nuevo":
            params["first_air_date.gte"] = "2015-01-01"
        elif novedad == "clasico":
            params["first_air_date.lte"] = "2005-12-31"

    # -----------------------------
    # Duración de película
    # -----------------------------
    if content_type == "movie":
        dur = slots.get("duracion_peli")
        if dur == "corta":
            params["with_runtime.lte"] = 100
        elif dur == "larga":
            params["with_runtime.gte"] = 130

    # -----------------------------
    # Filtro de fama
    # -----------------------------
    fama = slots.get("fama")
    if fama == "conocida":
        params["sort_by"] = "popularity.desc"
        params["vote_count.gte"] = 500
    elif fama == "joyita":
        params["sort_by"] = "vote_average.desc"
        params["vote_count.gte"] = 50
        params["vote_count.lte"] = 2000
    else:
        params["sort_by"] = "vote_average.desc"

    # -----------------------------
    # Construcción del endpoint final
    # -----------------------------
    if query_text:
        path = "/search/movie" if content_type == "movie" else "/search/tv"
        params["query"] = query_text
    else:
        path = "/discover/movie" if content_type == "movie" else "/discover/tv"

    # -----------------------------
    # Llamada final a TMDB
    # -----------------------------
    data = _tmdb_get(path, params)
    return data

# ------------------------------
# TMDB – plataformas en Argentina
# ------------------------------

def get_watch_providers(
    content_type: ContentType,
    tmdb_id: int,
    region: str | None = None,
) -> Dict[str, Any]:
    """
    Devuelve info de en qué plataformas se puede ver (flatrate, rent, buy)
    para una película o serie, filtrado por región (default: AR).
    """
    region = (region or settings.region or "AR").upper()

    data = _tmdb_get(f"/{content_type}/{tmdb_id}/watch/providers", {})
    results = data.get("results", {})
    region_info = results.get(region)

    if not region_info:
        return {
            "region": region,
            "available": False,
            "flatrate": [],
            "rent": [],
            "buy": [],
        }

    flatrate = region_info.get("flatrate") or []
    rent = region_info.get("rent") or []
    buy = region_info.get("buy") or []

    platforms_flatrate = [p["provider_name"] for p in flatrate if p.get("provider_name")]
    platforms_rent = [p["provider_name"] for p in rent if p.get("provider_name")]
    platforms_buy = [p["provider_name"] for p in buy if p.get("provider_name")]

    available = bool(platforms_flatrate or platforms_rent or platforms_buy)

    return {
        "region": region,
        "available": available,
        "flatrate": platforms_flatrate,
        "rent": platforms_rent,
        "buy": platforms_buy,
    }


def format_providers_message(providers: Dict[str, Any], content_type: ContentType) -> str:
    """
    Devuelve un texto lindo para el usuario sobre dónde ver la peli/serie en Argentina.
    """
    if not providers.get("available"):
        return "📍 No se encuentra disponible en plataformas de streaming en Argentina."

    flatrate = providers.get("flatrate") or []
    rent = providers.get("rent") or []
    buy = providers.get("buy") or []

    parts: List[str] = []

    if flatrate:
        parts.append("Incluida en suscripción en: " + ", ".join(flatrate))
    if rent:
        parts.append("Para alquilar en: " + ", ".join(rent))
    if buy:
        parts.append("Para comprar en: " + ", ".join(buy))

    if not parts:
        return "📍 No se encuentra disponible en plataformas de streaming en Argentina."

    label = "serie" if content_type == "tv" else "película"

    return f"📺 Esta {label} se puede ver en Argentina en:\n- " + "\n- ".join(parts)


# ------------------------------
# Recomendar a partir de resultados TMDB
# ------------------------------

def build_recommendations_from_tmdb(
    content_type: ContentType,
    tmdb_results: Dict[str, Any],
    slots: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    A partir de los resultados crudos de TMDB, arma una lista de recomendaciones
    con detalles y plataformas de Argentina.

    Devuelve una lista de dicts:
    {
      "id": int,
      "title": str,
      "overview": str,
      "genres": str,
      "year": str,
      "duration": str,        // runtime o duración por capítulo
      "seasons": int | None,  // solo series
      "episodes": int | None, // solo series
      "providers_text": str,  // texto ya listo para mostrar
    }
    """
    results = tmdb_results.get("results", []) or []
    if not results:
        return []

    max_recs = int(slots.get("cantidad_recs") or 1)
    max_recs = max(1, min(max_recs, 5))  # por las dudas

    recs: List[Dict[str, Any]] = []

    for item in results:
        tmdb_id = item.get("id")
        if not tmdb_id:
            continue

        # Detalles
        path = f"/movie/{tmdb_id}" if content_type == "movie" else f"/tv/{tmdb_id}"
        details = _tmdb_get(path, {"language": TMDB_LANG})

        if content_type == "movie":
            title = details.get("title") or details.get("original_title") or "Sin título"
            year = (details.get("release_date") or "")[:4] or "N/D"
            runtime = details.get("runtime")
            duration = f"{runtime} min" if runtime else "Duración N/D"
        else:
            title = details.get("name") or details.get("original_name") or "Sin título"
            year = (details.get("first_air_date") or "")[:4] or "N/D"
            runtimes = details.get("episode_run_time") or []
            duration = f"{runtimes[0]} min/episodio" if runtimes else "Duración N/D"

        overview = details.get("overview") or "Sin sinopsis disponible."
        genres_detail = details.get("genres") or []
        genres_text = ", ".join(g.get("name", "") for g in genres_detail[:3]) or "Género N/D"

        seasons = None
        episodes = None
        if content_type == "tv":
            seasons = details.get("number_of_seasons")
            episodes = details.get("number_of_episodes")

        providers = get_watch_providers(content_type, tmdb_id)
        providers_text = format_providers_message(providers, content_type)

        recs.append(
            {
                "id": tmdb_id,
                "title": title,
                "overview": overview,
                "genres": genres_text,
                "year": year,
                "duration": duration,
                "seasons": seasons,
                "episodes": episodes,
                "providers_text": providers_text,
            }
        )

        if len(recs) >= max_recs:
            break

    return recs
