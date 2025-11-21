# app/chat.py

from __future__ import annotations

from typing import Dict, Any, Tuple

import logging

from .utils import (
    extract_slots_from_text,
    merge_slots,
    discover_tmdb,
    build_recommendations_from_tmdb,
    save_conversation_history,
)

logger = logging.getLogger("moodflix")


class ChatManager:
    """
    Cerebro del bot: maneja el estado de la conversación por usuario,
    usa Groq para entender las preferencias (slots) y TMDB para encontrar
    la mejor peli/serie según lo que el usuario vaya respondiendo.
    """

    def __init__(self) -> None:
        self.conversation_state: Dict[str, Dict[str, Any]] = {}

    # -------------------------
    # Helpers de estado
    # -------------------------

    def _get_state(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.conversation_state:
            self.conversation_state[user_id] = {
                "slots": {},
                "last_intent": None,
                "last_question": None,
                "page": 1,
            }
        return self.conversation_state[user_id]

    def _reset_state(self, user_id: str) -> None:
        self.conversation_state[user_id] = {
            "slots": {},
            "last_intent": None,
            "last_question": None,
            "page": 1,
        }

    # -------------------------
    # 🔥 ESTE MÉTODO FALTABA ACÁ
    # -------------------------
    def _welcome_message(self) -> str:
        return (
            "👋 Hola, soy tu bot de recomendaciones de películas y series.\n\n"
            "Te voy a ir haciendo algunas preguntas para encontrar algo que encaje "
            "con lo que tenés ganas de ver.\n\n"
            "Podés empezar diciendo cosas como:\n"
            "• \"Recomendame una peli de terror cortita\"\n"
            "• \"Quiero una serie de comedia para ver con mi pareja\"\n"
        )

    # -------------------------
    # Punto de entrada
    # -------------------------

    def handle_message(self, user_id: str, text: str) -> str:
        raw_text = text.strip()
        lower = raw_text.lower()

        if lower in ("/start", "/reset", "/reiniciar"):
            self._reset_state(user_id)
            return self._welcome_message()

        if any(kw in lower for kw in ("hola", "holaa", "holis", "buenas", "buen dia", "buen día", "hey")):
            return self._welcome_message()

        if lower in ("otra", "otra peli", "otra película", "otra serie"):
            state = self._get_state(user_id)
            state["page"] += 1
            return self._try_recommend(user_id, reason="otra_opcion")

        if lower in ("chau", "chao", "me voy", "/end", "/stop"):
            self._reset_state(user_id)
            return "¡Gracias por usar el bot! Cuando quieras volvemos a buscar algo para ver 🍿"

        return self._process_user_message(user_id, raw_text)

    # -------------------------
    # Lógica principal
    # -------------------------

    def _process_user_message(self, user_id: str, text: str) -> str:
        state = self._get_state(user_id)
        slots_actuales = state["slots"]
        ultima_pregunta = state["last_question"]

        parsed = extract_slots_from_text(text)
        intent = parsed.get("intent", "other")
        new_slots = parsed.get("slots", {}) or {}

        if intent == "other" and ultima_pregunta is not None:
            intent = "answer"

        logger.info(
            "🧠 Parsed user (%s): intent=%s new_slots=%s",
            user_id, intent, new_slots
        )

        merged_slots = merge_slots(slots_actuales, new_slots)
        state["slots"] = merged_slots
        state["last_intent"] = intent
        state["page"] = 1

        if intent == "other":
            msg = (
                "No estoy segura de haber entendido 😅.\n"
                "Podés decirme cosas como:\n"
                "• \"Recomendame una película de comedia\"\n"
                "• \"Quiero una serie cortita para ver en familia\"\n"
            )
            save_conversation_history(user_id, text, msg, parsed)
            return msg

        question = self._next_question(merged_slots)
        if question:
            state["last_question"] = question["key"]
            reply = question["text"]
            save_conversation_history(user_id, text, reply, parsed)
            return reply

        reply = self._try_recommend(user_id)
        save_conversation_history(user_id, text, reply, parsed)
        return reply

    # -------------------------
    # Preguntas siguientes
    # -------------------------

    def _next_question(self, slots: Dict[str, Any]) -> Dict[str, str] | None:
        tipo = (slots.get("tipo_contenido") or "indiferente").lower()
        generos = slots.get("generos") or []
        novedad = slots.get("novedad")
        contexto = slots.get("contexto")
        fama = slots.get("fama")

        duracion_peli = slots.get("duracion_peli")
        temporadas = slots.get("temporadas")
        episodios_totales = slots.get("episodios_totales")
        duracion_capitulo = slots.get("duracion_capitulo")

        if tipo not in ("movie", "tv"):
            return {"key": "tipo_contenido",
                    "text": "¿Qué tenés ganas de ver ahora: **película**, **serie** o te da lo mismo?"}

        if not generos:
            return {"key": "generos",
                    "text": "Bien. ¿De qué onda te pinta?\nPodés decirme uno o varios géneros: comedia, terror, drama, acción, romántica, ciencia ficción, etc."}

        if tipo == "movie" and duracion_peli in (None, "", "indiferente"):
            return {"key": "duracion_peli",
                    "text": "¿Buscás una **peli cortita** o una **larga**?"}

        if tipo == "tv":
            if temporadas in (None, "", "indiferente"):
                return {"key": "temporadas", "text": "¿Pocas temporadas o varias?"}
            if episodios_totales in (None, "", "indiferente"):
                return {"key": "episodios_totales",
                        "text": "¿Pocos capítulos o muchos?"}
            if duracion_capitulo in (None, "", "indiferente"):
                return {"key": "duracion_capitulo",
                        "text": "¿Capítulos cortitos (20-30 min) o largos (40-60 min)?"}

        if novedad in (None, "", "indiferente"):
            return {"key": "novedad",
                    "text": "¿Preferís algo **nuevo** o también te va algún **clásico**?"}

        if contexto in (None, ""):
            return {"key": "contexto",
                    "text": "¿Lo vas a ver solo, con pareja, con amigxs o en familia?"}

        if fama in (None, "", "indiferente"):
            return {"key": "fama",
                    "text": "¿Algo muy conocido o una joyita poco vista?"}

        return None

    # -------------------------
    # Recomendaciones
    # -------------------------

    def _try_recommend(self, user_id: str, reason: str = "normal") -> str:
        state = self._get_state(user_id)
        slots = state["slots"]
        page = state["page"]

        tipo = (slots.get("tipo_contenido") or "movie").lower()
        if tipo not in ("movie", "tv"):
            tipo = "movie"

        logger.info(f"🎯 Recomendar para user={user_id} tipo={tipo} slots={slots} page={page}")

        try:
            tmdb_results = discover_tmdb(tipo, slots, page)
        except Exception as e:
            logger.error(f"❌ Error TMDB: {e}")
            return "Error con la API, probá en un ratito."

        recs = build_recommendations_from_tmdb(tipo, tmdb_results, slots)

        if not recs:
            return "Con lo que me contaste no encontré nada 😕. Probá cambiando algún filtro."

        parts = []
        intro = "Te dejo una recomendación" if len(recs) == 1 else "Mirá estas recomendaciones"
        if reason == "otra_opcion":
            intro = "Te dejo otra opción:"

        parts.append(intro + " 👇\n")

        for rec in recs:
            title = f"🎬 *{rec['title']}* ({rec['year']})"
            genres = f"• Géneros: {rec['genres']}"
            duration = f"• Duración: {rec['duration']}"

            extras = []
            if tipo == "tv":
                if rec.get("seasons") is not None:
                    extras.append(f"• Temporadas: {rec['seasons']}")
                if rec.get("episodes") is not None:
                    extras.append(f"• Episodios: {rec['episodes']}")

            overview = rec["overview"]
            if len(overview) > 380:
                overview = overview[:380].rsplit(" ", 1)[0] + "..."

            providers = rec["providers_text"]

            block = [
                title, genres, duration, *extras, "",
                f"📝 {overview}", "",
                providers,
                ""
            ]
            parts.append("\n".join(block))

        parts.append('Decime *"otra"* para más opciones o cambiá algún filtro.')

        return "\n".join(parts).strip()
