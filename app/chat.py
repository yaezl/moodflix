# app/chat.py
from typing import Dict, Any, List
import logging
import time

from .utils import (
    parse_user_intent_with_openai,
    detect_strategy_from_text,
    infer_strategy_with_openai,
    save_conversation_history,
    get_movie_recommendations,
    get_series_recommendations,
    get_music_recommendations,
)

logger = logging.getLogger("moodflix")


class RecommenderChatBot:
    def __init__(self, settings):
        self.settings = settings
        # Estado de la conversación
        self.pending_intents: dict[str, dict] = {}
        self.waiting_for: dict[str, str] = {}
        # MEMORIA: última recomendación por usuario
        self.last_recommendation: dict[str, dict] = {}
        self.last_activity: dict[str, float] = {}
        # TRACKING: IDs ya recomendados por usuario
        self.recommended_ids: dict[str, set] = {}
        # Página actual para paginación
        self.current_page: dict[str, int] = {}

    def _build_recommendation_text(self, parsed: dict, user_id: str = None) -> str:
        tipo = parsed.get("type")
        mood = parsed.get("mood")
        strategy = parsed.get("match_strategy")
        genre_name = parsed.get("genre")

        if strategy == "match":
            strategy_text = "acompañar tu vibe"
        else:
            strategy_text = "cambiar tu mood"

        genre_label = f" de *{genre_name}*" if genre_name else ""

        # Obtener IDs ya vistos
        seen_ids = self.recommended_ids.get(user_id, set()) if user_id else set()

        # PELÍCULAS
        if tipo == "movie":
            recs = get_movie_recommendations(parsed, seen_ids=seen_ids)
            if not recs:
                # Si no hay más, resetear y buscar de nuevo
                if user_id and seen_ids:
                    self.recommended_ids[user_id] = set()
                    recs = get_movie_recommendations(parsed, seen_ids=set())
                
                if not recs:
                    return "Mmm, no encontré pelis 😕\nRevisá tu API key o probá otra búsqueda."

            # Guardar IDs recomendados
            if user_id:
                if user_id not in self.recommended_ids:
                    self.recommended_ids[user_id] = set()
                for r in recs:
                    self.recommended_ids[user_id].add(r.get("id"))

            if genre_name and (mood == "neutral" or not mood):
                title_line = f"🎬 *Pelis{genre_label}:*"
            else:
                title_line = f"🎬 *Pelis{genre_label} para {strategy_text}:*"

            lines: List[str] = [title_line, ""]
            for i, r in enumerate(recs, start=1):
                overview = r["overview"][:180] + "..." if len(r["overview"]) > 180 else r["overview"]

                lines.append(
                    f"{i}. *{r['title']}* ({r['year']})\n"
                    f"   📁 {r['genre']} · ⏱ {r['duration']}\n"
                    f"   📺 {r['platforms']}\n"
                    f"   _{overview}_\n"
                )

            return "\n".join(lines)

        # SERIES
        if tipo == "series":
            recs = get_series_recommendations(parsed, seen_ids=seen_ids)
            if not recs:
                if user_id and seen_ids:
                    self.recommended_ids[user_id] = set()
                    recs = get_series_recommendations(parsed, seen_ids=set())
                
                if not recs:
                    return "No encontré series 😕\nRevisá tu API key o probá otra búsqueda."

            if user_id:
                if user_id not in self.recommended_ids:
                    self.recommended_ids[user_id] = set()
                for r in recs:
                    self.recommended_ids[user_id].add(r.get("id"))

            if genre_name and (mood == "neutral" or not mood):
                title_line = f"📺 *Series{genre_label}:*"
            else:
                title_line = f"📺 *Series{genre_label} para {strategy_text}:*"

            lines: List[str] = [title_line, ""]
            for i, r in enumerate(recs, start=1):
                overview = r["overview"][:180] + "..." if len(r["overview"]) > 180 else r["overview"]

                lines.append(
                    f"{i}. *{r['title']}* ({r['year']})\n"
                    f"   📁 {r['genre']} · 📊 {r['seasons']} temp.\n"
                    f"   📺 {r['platforms']}\n"
                    f"   _{overview}_\n"
                )

            return "\n".join(lines)

        # MÚSICA
        if tipo == "music":
            recs = get_music_recommendations(parsed, seen_ids=seen_ids)
            if not recs:
                if user_id and seen_ids:
                    self.recommended_ids[user_id] = set()
                    recs = get_music_recommendations(parsed, seen_ids=set())
                
                if not recs:
                    return "No encontré música 😕\nRevisá tus credenciales de Spotify."

            if user_id:
                if user_id not in self.recommended_ids:
                    self.recommended_ids[user_id] = set()
                for r in recs:
                    self.recommended_ids[user_id].add(r.get("id"))

            if genre_name:
                header = f"🎧 *{genre_name.title()}*"
            else:
                header = "🎧 *Tu playlist*"

            if mood and mood != "neutral":
                header += f" · {strategy_text}"

            lines: List[str] = [header, ""]

            for i, r in enumerate(recs, start=1):
                lines.append(
                    f"{i}. *{r['title']}*\n"
                    f"   🎤 {r['artist']}\n"
                    f"   🔗 {r['url']}\n"
                )

            return "\n".join(lines)

        return "Mmm, algo se mezcló 😅 Probá de nuevo."

    def handle_message(self, user_id: str, text: str) -> str:
        raw = text.strip()
        lower = raw.lower()

        logger.info(f"💬 Usuario {user_id}: '{raw}'")

        # Limpiar actividad vieja (más de 5 minutos)
        now = time.time()
        if user_id in self.last_activity:
            if now - self.last_activity[user_id] > 300:
                self.pending_intents.pop(user_id, None)
                self.waiting_for.pop(user_id, None)
                self.last_recommendation.pop(user_id, None)
        
        self.last_activity[user_id] = now

        # SALUDOS
        if any(g in lower for g in ["hola", "holis", "buenas", "buen día", "hey", "hi"]):
            self.pending_intents.pop(user_id, None)
            self.waiting_for.pop(user_id, None)
            self.last_recommendation.pop(user_id, None)
            self.recommended_ids.pop(user_id, None)  # Limpiar historial
            
            return (
                "¡Hola! 👋 Soy *MoodFlix*\n\n"
                "Te recomiendo:\n"
                "🎬 Películas · 📺 Series · 🎧 Música\n\n"
                "Contame qué onda 😊"
            )

        # COMANDOS DE CONTINUACIÓN
        if any(w in lower for w in ["mas", "más", "otra", "otro", "dame mas", "dame más"]):
            if user_id in self.last_recommendation:
                last = self.last_recommendation[user_id]
                response_text = self._build_recommendation_text(last, user_id)
                save_conversation_history(user_id, raw, response_text, last)
                return response_text
            else:
                return "No tengo un pedido anterior 🤔\nDecime qué querés."

        # CAMBIO DE TIPO ("ahora pelis", "ahora series")
        tipo_change = None
        if any(w in lower for w in ["ahora peli", "ahora una peli", "y peli"]):
            tipo_change = "movie"
        elif any(w in lower for w in ["ahora serie", "ahora una serie", "y serie"]):
            tipo_change = "series"
        elif any(w in lower for w in ["ahora música", "ahora musica", "y música", "y musica"]):
            tipo_change = "music"

        if tipo_change and user_id in self.last_recommendation:
            last = self.last_recommendation[user_id].copy()
            last["type"] = tipo_change
            # Resetear IDs vistos al cambiar de tipo
            self.recommended_ids.pop(user_id, None)
            self.last_recommendation[user_id] = last
            response_text = self._build_recommendation_text(last, user_id)
            save_conversation_history(user_id, raw, response_text, last)
            return response_text

        # FLUJO CON ESTADO PENDIENTE
        if user_id in self.waiting_for:
            waiting = self.waiting_for[user_id]
            parsed = self.pending_intents[user_id]

            # ESPERANDO TIPO
            if waiting == "type":
                new_parsed = parse_user_intent_with_openai(raw)
                new_type = new_parsed.get("type")
                
                if new_type and new_type != "unknown":
                    parsed["type"] = new_type
                    mood = parsed.get("mood")
                    genre = parsed.get("genre")
                    
                    if genre and (not mood or mood == "neutral"):
                        parsed["match_strategy"] = "match"
                        del self.waiting_for[user_id]
                        del self.pending_intents[user_id]
                        self.last_recommendation[user_id] = parsed
                        
                        response_text = self._build_recommendation_text(parsed, user_id)
                        save_conversation_history(user_id, raw, response_text, parsed)
                        return response_text
                    
                    if mood and mood != "neutral":
                        self.waiting_for[user_id] = "strategy"
                        
                        tipo_texto = {"music": "música", "movie": "una peli", "series": "una serie"}.get(new_type, "contenido")
                        
                        response_text = (
                            f"Dale! {tipo_texto} para tu vibe *{mood}* 😊\n\n"
                            "¿Querés que *acompañe* o *cambie* tu mood?"
                        )
                        save_conversation_history(user_id, raw, response_text, parsed)
                        return response_text
                    
                    parsed["match_strategy"] = "match"
                    del self.waiting_for[user_id]
                    del self.pending_intents[user_id]
                    self.last_recommendation[user_id] = parsed
                    
                    response_text = self._build_recommendation_text(parsed)
                    save_conversation_history(user_id, raw, response_text, parsed)
                    return response_text
                else:
                    return "No caché si querés peli, serie o música 🤔"

            # ESPERANDO MOOD
            elif waiting == "mood":
                new_parsed = parse_user_intent_with_openai(raw)
                new_mood = new_parsed.get("mood")
                
                if new_mood and new_mood != "neutral":
                    parsed["mood"] = new_mood
                    self.waiting_for[user_id] = "strategy"
                    
                    response_text = (
                        f"Dale, estás *{new_mood}* 😊\n\n"
                        "¿Querés que *acompañe* o *cambie* tu mood?"
                    )
                    save_conversation_history(user_id, raw, response_text, parsed)
                    return response_text
                else:
                    parsed["mood"] = "neutral"
                    parsed["match_strategy"] = "match"
                    del self.waiting_for[user_id]
                    del self.pending_intents[user_id]
                    self.last_recommendation[user_id] = parsed
                    
                    response_text = self._build_recommendation_text(parsed)
                    save_conversation_history(user_id, raw, response_text, parsed)
                    return response_text

            # ESPERANDO ESTRATEGIA
            elif waiting == "strategy":
                strategy = detect_strategy_from_text(raw)
                
                new_parsed = parse_user_intent_with_openai(raw)
                new_mood = new_parsed.get("mood")
                if new_mood and new_mood != "neutral":
                    parsed["mood"] = new_mood

                if not strategy:
                    return (
                        "No caché si querés que *acompañe* o *cambie* 🤔\n"
                        "Decime: \"que acompañe\" o \"que lo cambie\""
                    )

                parsed["match_strategy"] = strategy
                del self.waiting_for[user_id]
                del self.pending_intents[user_id]
                self.last_recommendation[user_id] = parsed

                response_text = self._build_recommendation_text(parsed)
                save_conversation_history(user_id, raw, response_text, parsed)
                return response_text

        # NUEVO PEDIDO
        parsed = parse_user_intent_with_openai(raw)
        tipo = parsed.get("type")
        mood = parsed.get("mood")
        genre_name = parsed.get("genre")

        logger.info(f"🧠 Parse: type={tipo}, mood={mood}, genre={genre_name}")

        # CASO 1: Solo género
        if genre_name and (not mood or mood == "neutral") and tipo in ("movie", "series", "music"):
            parsed["match_strategy"] = "match"
            self.last_recommendation[user_id] = parsed
            response_text = self._build_recommendation_text(parsed)
            save_conversation_history(user_id, raw, response_text, parsed)
            return response_text

        # CASO 2: Tipo + mood
        if tipo in ("movie", "series", "music") and mood and mood != "neutral":
            strategy = detect_strategy_from_text(raw)
            
            if strategy:
                parsed["match_strategy"] = strategy
                self.last_recommendation[user_id] = parsed
                response_text = self._build_recommendation_text(parsed)
                save_conversation_history(user_id, raw, response_text, parsed)
                return response_text
            else:
                self.pending_intents[user_id] = parsed
                self.waiting_for[user_id] = "strategy"

                tipo_texto = {"music": "música", "movie": "una peli", "series": "una serie"}.get(tipo, "contenido")

                response_text = (
                    f"Dale! {tipo_texto} con vibe *{mood}* 😊\n\n"
                    "¿Querés que *acompañe* o *cambie* tu mood?"
                )
                save_conversation_history(user_id, raw, response_text, parsed)
                return response_text

        # CASO 3: Solo tipo
        if tipo in ("movie", "series", "music") and (not mood or mood == "neutral"):
            self.pending_intents[user_id] = parsed
            self.waiting_for[user_id] = "mood"

            tipo_texto = {"music": "música", "movie": "una peli", "series": "una serie"}.get(tipo, "contenido")

            response_text = (
                f"Dale! Querés {tipo_texto} 😊\n\n"
                "Contame, ¿cómo te sentís o qué estás haciendo?"
            )
            save_conversation_history(user_id, raw, response_text, parsed)
            return response_text

        # CASO 4: Solo mood
        if tipo == "unknown" and mood and mood != "neutral":
            self.pending_intents[user_id] = parsed
            self.waiting_for[user_id] = "type"

            response_text = (
                f"Dale, estás *{mood}* 😊\n\n"
                "¿Qué querés?\n"
                "🎬 Peli · 📺 Serie · 🎧 Música"
            )
            save_conversation_history(user_id, raw, response_text, parsed)
            return response_text

        # CASO 5: No entendió
        return (
            "No caché qué querés 🤔\n\n"
            "Probá:\n"
            "• \"Pelis de terror\"\n"
            "• \"Música para correr\"\n"
            "• \"Estoy triste, pasame una serie\""
        )