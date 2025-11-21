# app/utils.py

from typing import Dict, Any, List, Literal
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

def groq_chat(system_prompt: str, user_prompt: str, temperature: float = 0.6) -> str:
    """
    Llama al modelo de Groq y devuelve el contenido de la respuesta.
    Usalo cuando querés una respuesta en texto libre.
    """
    logger.debug("🤖 GROQ chat → %s...", user_prompt[:60])

    resp = groq_client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    content = content.strip()
    logger.debug("🤖 GROQ chat ← %s...", content[:60])
    return content


def groq_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Igual que groq_chat, pero asumiendo que el modelo responde SOLO JSON.
    Si falla el parseo, devuelve {}.
    Además, limpia fences tipo ```json ... ``` que a veces agrega el modelo.
    """
    content = groq_chat(system_prompt, user_prompt, temperature=0.0)

    # Limpieza básica de fences ```json ... ```
    cleaned = content.strip()

    # Caso: ```json\n{ ... }\n```
    if cleaned.startswith("```"):
        # sacamos el bloque exterior de ```
        # nos quedamos con lo de adentro
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)   # saca ``` o ```json al principio
        cleaned = re.sub(r"\s*```$", "", cleaned)            # saca ``` al final

    try:
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError:
        # si sigue fallando, avisamos pero recortando el chorizo
        preview = cleaned.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        logger.warning("⚠️ No se pudo parsear JSON desde Groq. Respuesta (inicio): %s", preview)
        return {}

    """
    Igual que groq_chat, pero asumiendo que el modelo responde SOLO JSON.
    Si falla el parseo, devuelve {}.
    """
    content = groq_chat(system_prompt, user_prompt, temperature=0.0)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("⚠️ No se pudo parsear JSON desde Groq. Respuesta: %s", content)
        return {}


from typing import Dict, Any, Optional
import json
import re
...

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

    system_prompt = f"""
Sos un asistente que SOLO devuelve JSON con este formato:

{{
  "intent": "recommendation" | "answer" | "other",
  "slots": {{
    "tipo_contenido": "movie" | "tv" | "indiferente" | null,
    "generos": [ "comedia", "terror", "drama", ... ],
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
- Última pregunta que se le hizo al usuario (en texto humano, no clave interna): "{last_question}".
- Slots actuales (lo que ya sabemos): {json.dumps(prev_slots, ensure_ascii=False)}.

REGLAS IMPORTANTES:

1. Si el usuario responde con algo muy corto (por ejemplo "pocas", "varias", "largos", "cortos",
   "conocida", "clásico", "nuevo", "con amigxs", etc.), interpretalo como
   RESPUESTA DIRECTA a la última pregunta.

   Ejemplos:
   - Pregunta: "¿Pocas temporadas o varias?" → usuario: "pocas" → "temporadas": "pocas"
   - Pregunta: "¿Pocos capítulos o muchos?" → usuario: "pocos" → "episodios_totales": "pocos"
   - Pregunta: "Capítulos cortitos (20-30 min) o largos (40-60 min)?" → "largos" → "duracion_capitulo": "largos"
   - Pregunta: "¿Algo muy conocido o una joyita poco vista?" → "bastante conocido" → "fama": "conocida"
   - Pregunta: "¿Lo vas a ver solo, con pareja, con amigxs o en familia?" → "eee con amigxs" → "contexto": "amigxs"
   - Pregunta: "¿Preferís algo nuevo o también te va algún clásico?" → "algo clásico" → "novedad": "clasico"

2. Aceptá SINÓNIMOS en español:
   - "pocas", "una temporada", "cortita en temporadas" → "pocas"
   - "varias", "muchas", "larguita de temporadas" → "varias"
   - "cortos", "cortitos", "episodios cortos", "capítulos cortos" → "cortos"
   - "largos", "capítulos largos", "episodios largos" → "largos"
   - "nuevo", "moderno", "de ahora", "reciente", "actual" → "nuevo"
   - "clásico", "viejo", "antiguo pero bueno" → "clasico"
   - "conocida", "popular", "famosa", "muy conocida" → "conocida"
   - "joyita", "poco conocida", "joya oculta", "desconocida" → "joyita"

3. Si ya sabemos el tipo de contenido en slots anteriores (por ejemplo "movie" o "tv"),
   usalo como contexto para interpretar la respuesta del usuario.

4. "intent":
   - "recommendation": cuando el usuario pide que le recomiendes algo o cambia de tipo de contenido
     (ej: "recomendame una peli", "ahora quiero una serie").
   - "answer": cuando está respondiendo una de tus preguntas (duración, temporadas, etc.).
   - "other": cuando habla de algo que no tiene que ver con la recomendación.

5. Interpretación de indiferencia:

Cuando el usuario diga cosas como: 
"me da igual", "indiferente", "no sé", "nose", "cualquiera", 
"como quieras", "lo que vos digas", "no tengo preferencia", 
"sin preferencia", "mmm cualquiera", "mmm da igual", 
"ninguna preferencia", "me es indistinto", "da igual", 
interpretalo como RESPUESTA DIRECTA a la última pregunta.

En ese caso, asigná el slot correspondiente a:
"indiferente"

Ejemplos concretos:
- Pregunta: "¿Pocas temporadas (1–3) o varias (4+)?"
  Usuario: "me da igual" 
  → temporadas: "indiferente"

- Pregunta: "¿Pocos capítulos (menos de 30) o muchos (30+)?"
  Usuario: "cualquiera" 
  → episodios_totales: "indiferente"

- Pregunta: "¿Capítulos cortitos (20–30 min) o largos (40–60 min)?"
  Usuario: "no sé"
  → duracion_capitulo: "indiferente"

- Pregunta: "¿Preferís algo nuevo o un clásico?"
  Usuario: "como quieras"
  → novedad: "indiferente"

- Pregunta: "¿Algo muy conocido o una joyita?"
  Usuario: "mmm cualquiera"
  → fama: "indiferente"   

6. Interpretación de temporadas:
- "pocas", "1 temporada", "una temporada", "dos temporadas", "tres", 
  "entre 1 y 3", "1–3", "temporadas cortas", 
  "poquitas temporadas" 
  → temporadas: "pocas"

- "varias", "muchas", "4 temporadas", "más de tres", "4 o más", "4+", 
  "temporadas largas", "bocha de temporadas", 
  "varias temporadas"
  → temporadas: "varias"

7. Interpretación de cantidad total de capítulos:
- "pocos capítulos", "menos de 30", "serie cortita", "pocos episodios",
  "rápida de ver", "liviana", "capítulos en total pocos"
  → episodios_totales: "pocos"

- "muchos capítulos", "más de 30", "bocha de episodios", 
  "larga para engancharse", "muchos episodios",
  "capítulos en total muchos"
  → episodios_totales: "muchos"

8. Interpretación de duración por capítulo:
- "cortos", "cortitos", "20 minutos", "media hora", 
  "capítulos chicos", "rápidos"
  → duracion_capitulo: "cortos"

- "largos", "45 minutos", "una hora", "capítulos largos",
  "episodios largos", "capítulos de una hora"
  → duracion_capitulo: "largos"

9. Interpretación de restricciones:

Estas frases deben mapearse al campo "restricciones" y devolver valores
estandarizados en forma de lista, por ejemplo:
"restricciones": ["no_animacion"]

    1) No animación:
    Frases como:
    "no animada", "que no sea animada", 
    "no de animación", "sin animación", 
    "no dibujitos", "no infantil"
    → restricciones: ["no_animacion"]

    2) No terror / no sustos:
    "no terror", "que no sea de terror", 
    "no cosas que asusten", "no sustos", 
    "no quiero nada de miedo", "sin miedo"
    → restricciones: ["no_terror"]

    3) No gore / no violencia / no sangrienta:
    "sin gore", "no gore", 
    "no muy fuerte", "no muy violenta",
    "no sangrienta", "no sangre", 
    "no violencia fuerte"
    → restricciones: ["no_gore"]

    4) No romance:
    "no romántica", "sin romance", 
    "no algo cursi", "odio el romance"
    → restricciones: ["no_romance"]

    5) No ciencia ficción / no fantasía:
    "no sci fi", "no ciencia ficción",
    "no cosas futuristas",
    "no fantasía", "sin magia"
    → restricciones: ["no_scifi"]

    6) No crimen / no policiales:
    "no policiales", "no crimen",
    "no detectivesco"
    → restricciones: ["no_crimen"]

    7) No bélicas:
    "no guerra", "no belicas",
    "no militares"
    → restricciones: ["no_guerra"]

IMPORTANTE:
- Las restricciones deben ser una lista.
- Si el usuario menciona más de una restricción, deben combinarse.
- Si responde algo tipo "me da igual" o "cualquiera", NO agregues restricciones.

10. Interpretación de contexto social:

- "solo", "sola", "solito", "para ver solo" 
  → contexto: "solo"

- "pareja", "mi novio", "mi novia", "mi pareja", "con mi pareja"
  → contexto: "pareja"

- "amigxs", "mis amigas", "con amigos", "con mis amigos", "con amigxs"
  → contexto: "amigxs"

- "familia", "familiar", "para ver con mi familia"
  → contexto: "familia"

11. Interpretación de temáticas (slot "tematicas"):

Usá el slot "tematicas" para cosas más específicas que el género:
ejemplos: sobrenatural, vampiros, hombres lobo, doctores, abogados, guerra,
amistad, carreras, basada en hechos reales, etc.

Mapeá expresiones del usuario a valores normalizados (snake_case) en "tematicas":

- "sobrenatural", "cosas sobrenaturales", "algo sobrenatural"
  → tematicas: ["sobrenatural"]

- "de vampiros", "sobre vampiros", "con vampiros", "vampiros y sangre"
  → tematicas: ["vampiros"]

- "de hombres lobo", "hombres lobos", "werewolf"
  → tematicas: ["hombres_lobo"]

- "de doctores", "de médicos", "hospitales", "médicos en hospital"
  → tematicas: ["doctores"]

- "de abogados", "juicios", "tribunales", "bufete de abogados"
  → tematicas: ["abogados"]

- "de guerra", "sobre la guerra", "bélica realista"
  → tematicas: ["guerra"]

- "de amigos", "sobre amistad", "grupo de amigos"
  → tematicas: ["amigos"]

- "de carreras", "carreras de autos", "racing", "coches de carrera"
  → tematicas: ["carreras_autos"]

- "basada en hechos reales", "basada en una historia real",
  "inspirada en hechos reales"
  → tematicas: ["hechos_reales"]

Si el usuario menciona varias cosas, combiná en la lista, por ejemplo:
"una peli de guerra basada en hechos reales"
→ tematicas: ["guerra", "hechos_reales"]


Devolvé SIEMPRE solo el JSON, sin texto adicional ni ```.
"""

    user_prompt = f"Mensaje del usuario: {user_text}"

    data = groq_json(system_prompt, user_prompt)

    # Fallback seguro
    intent = data.get("intent", "other")
    slots = data.get("slots", {}) or {}

    # Aseguramos campos mínimos
    if "cantidad_recs" not in slots:
        slots["cantidad_recs"] = 1

    return {"intent": intent, "slots": slots}


def merge_slots(current: Dict[str, Any], new_slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mezcla preferencias nuevas con las ya existentes.
    Si en new_slots hay un valor no vacío/distinto de 'indiferente', pisa al actual.
    """
    if not current:
        return new_slots.copy()

    merged = current.copy()

    for key, value in new_slots.items():
        if value in (None, "", [], "indiferente"):
            continue
        merged[key] = value

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
    
    # Mapeo de restricciones a IDs de géneros TMDB
    restriccion_a_generos = {
        "no_animacion": ["16"],
        "no_terror": ["27"],
        "no_romance": ["10749"],
        "no_scifi": ["878", "14"],
        "no_crimen": ["80"],
        "no_guerra": ["10752"],
    }

    # Armamos parámetro without_genres si corresponde
    without = []

    for r in restricciones:
        if r in restriccion_a_generos:
            without.extend(restriccion_a_generos[r])

    if without:
        params["without_genres"] = ",".join(without)

    params: Dict[str, Any] = {
        "language": TMDB_LANG,
        "region": settings.region.upper(),
        "include_adult": "false",
        "page": page,
    }

    # Si el usuario pidió "no animación", excluimos el género Animation (id 16 en TMDB)
    if "no_animacion" in restricciones:
        params["without_genres"] = "16"

    # Géneros
    generos = slots.get("generos") or []
    genre_ids = _resolve_genre_ids(content_type, generos)
    if genre_ids:
        params["with_genres"] = ",".join(str(g) for g in genre_ids)

    # Novedad
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

    # Duración película
    if content_type == "movie":
        dur = slots.get("duracion_peli")
        if dur == "corta":
            params["with_runtime.lte"] = 100
        elif dur == "larga":
            params["with_runtime.gte"] = 130

    # Fama (popularidad)
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

    # Llamar a TMDB
    path = "/discover/movie" if content_type == "movie" else "/discover/tv"
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
