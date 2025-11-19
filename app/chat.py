# app/chat.py
from typing import Dict, Any, List

from .utils import (
    parse_user_intent_with_openai,
    detect_strategy_from_text,
    infer_strategy_with_openai,
    save_conversation_history,
    get_movie_recommendations,
    get_series_recommendations,
    get_music_recommendations,
)


class RecommenderChatBot:
    def __init__(self, settings):
        self.settings = settings
        # Guardamos intención pendiente por usuario: { user_id: parsed_intent }
        self.pending_intents: dict[str, dict] = {}

    # ---------------------------
    # Armado de respuestas finales
    # ---------------------------
    def _build_recommendation_text(self, parsed: dict) -> str:
        tipo = parsed.get("type")
        mood = parsed.get("mood")
        strategy = parsed.get("match_strategy")
        genre_name = parsed.get("genre")

        # Texto para estrategia
        if strategy == "match":
            strategy_text = "acompañar tu estado"
        else:
            strategy_text = "cambiar tu estado / levantarte"

        # Si viene género explícito, lo usamos en el encabezado
        genre_label = f" de **{genre_name}**" if genre_name else ""

        # PELÍCULAS
        if tipo == "movie":
            recs = get_movie_recommendations(parsed)
            if not recs:
                return (
                    "Intenté buscar películas pero no encontré resultados o hubo un problema con TMDB.\n"
                    "Verificá tu TMDB_API_KEY o probá describiéndome de otra forma qué querés ver."
                )

            # Encabezado depende de si hay mood o solo género
            if genre_name and (mood == "neutral" or not mood):
                title_line = f"🎬 Te recomiendo estas películas{genre_label}:"
            else:
                title_line = (
                    f"🎬 Te recomiendo estas películas{genre_label} "
                    f"para {strategy_text} estando **{mood}**:"
                )

            lines: List[str] = [title_line]
            for i, r in enumerate(recs, start=1):
                overview = r["overview"]
                if len(overview) > 220:
                    overview = overview[:220] + "…"

                lines.append(
                    f"\n{i}. **{r['title']}** ({r['year']})\n"
                    f"   Género: {r['genre']}\n"
                    f"   Duración: {r['duration']}\n"
                    f"   Plataformas (AR): {r['platforms']}\n"
                    f"   Sinopsis: {overview}"
                )

            return "\n".join(lines)

        # SERIES
        if tipo == "series":
            recs = get_series_recommendations(parsed)
            if not recs:
                return (
                    "Intenté buscar series pero no encontré resultados o hubo un problema con TMDB.\n"
                    "Verificá tu TMDB_API_KEY o probá describiéndome de otra forma qué querés ver."
                )

            if genre_name and (mood == "neutral" or not mood):
                title_line = f"📺 Te recomiendo estas series{genre_label}:"
            else:
                title_line = (
                    f"📺 Te recomiendo estas series{genre_label} "
                    f"para {strategy_text} estando **{mood}**:"
                )

            lines: List[str] = [title_line]
            for i, r in enumerate(recs, start=1):
                overview = r["overview"]
                if len(overview) > 220:
                    overview = overview[:220] + "…"

                lines.append(
                    f"\n{i}. **{r['title']}** ({r['year']})\n"
                    f"   Género: {r['genre']}\n"
                    f"   Temporadas: {r['seasons']}, episodios: {r['episodes']}\n"
                    f"   Duración: {r['duration']}\n"
                    f"   Plataformas (AR): {r['platforms']}\n"
                    f"   Sinopsis: {overview}"
                )

            return "\n".join(lines)

        # MÚSICA
        if tipo == "music":
            recs = get_music_recommendations(parsed)
            genre_name = parsed.get("genre")
            if not recs:
                return (
                    "Intenté buscar música en Spotify pero no encontré resultados o hubo un problema de conexión.\n"
                    "Verificá tus credenciales de Spotify o probá describiéndome de otra forma qué querés escuchar."
                )

            if genre_name:
                header = f"🎧 Te recomiendo estas canciones de **{genre_name}** "
            else:
                header = "🎧 Te recomiendo estas canciones "

            if mood and mood != "neutral":
                header += f"para {strategy_text} estando **{mood}**:"
            else:
                header += ":"

            lines: List[str] = [header]

            for i, r in enumerate(recs, start=1):
                lines.append(
                    f"\n{i}. **{r['title']}** – {r['artist']}\n"
                    f"   Género(s): {r['genres']}\n"
                    f"   Escuchar en Spotify: {r['url']}"
                )

            return "\n".join(lines)

        return (
            "Se me mezcló un poco el contexto 😅, probá pidiéndome de nuevo música, película o serie."
        )

    # ---------------------------
    # Lógica principal del bot
    # ---------------------------
    def handle_message(self, user_id: str, text: str) -> str:
        # 1) Si ya teníamos una intención pendiente, este mensaje es la ESTRATEGIA
        if user_id in self.pending_intents:
            parsed = self.pending_intents[user_id]

            # Intentamos inferir estrategia (match/contrast)
            strategy = infer_strategy_with_openai(text, parsed)
            if strategy is None:
                strategy = detect_strategy_from_text(text)

            # Además, si el usuario corrige el mood (ej: "no estoy relajada, estoy estresada"),
            # volvemos a interpretar y si cambia el mood lo actualizamos.
            new_parsed = parse_user_intent_with_openai(text)
            new_mood = new_parsed.get("mood")
            if new_mood and new_mood != "neutral" and new_mood != parsed.get("mood"):
                parsed["mood"] = new_mood

            if strategy is None:
                response_text = (
                    "No terminé de entender si querés que las recomendaciones **acompañen** "
                    "tu estado de ánimo o que lo **cambien**.\n\n"
                    "Podés responder algo como:\n"
                    "- \"Que acompañe\" / \"que siga igual\"\n"
                    "- \"Que cambie mi ánimo\" / \"algo para levantarme\""
                )
                save_conversation_history(user_id, text, response_text, parsed)
                return response_text

            parsed["match_strategy"] = strategy
            del self.pending_intents[user_id]

            response_text = self._build_recommendation_text(parsed)
            save_conversation_history(user_id, text, response_text, parsed)
            return response_text

        # 2) NUEVO PEDIDO
        parsed = parse_user_intent_with_openai(text)
        tipo = parsed["type"]
        mood = parsed["mood"]
        genre_name = parsed.get("genre")

        # Si es un pedido por género (películas de terror, series de comedia, música pop, rock, etc.)
        # y no hay mood fuerte, NO preguntamos nada → devolvemos directo.
        if genre_name and (mood == "neutral" or not mood) and tipo in ("movie", "series", "music"):
            parsed["match_strategy"] = "match"  # default razonable
            response_text = self._build_recommendation_text(parsed)
            save_conversation_history(user_id, text, response_text, parsed)
            return response_text

        # Si no, flujo normal: ver si ya dijo match/contrast en el mismo mensaje
        strategy = detect_strategy_from_text(text)

        if strategy is None:
            # Preguntamos match vs cambiar estado y guardamos intención pendiente
            self.pending_intents[user_id] = parsed

            response_text = (
                "Ok, entendí que querés "
                f"{'música' if tipo == 'music' else 'una película' if tipo == 'movie' else 'una serie'} "
                f"y tu estado/mood se siente más bien **{mood}**.\n\n"
                "¿Querés que las recomendaciones **acompañen** tu estado de ánimo/actividad "
                "(match) o que lo **cambien** (contraste, algo para levantarte/animarte)?\n"
                "Podés responder algo como:\n"
                "- \"Que acompañe\" / \"que siga igual\"\n"
                "- \"Que cambie mi ánimo\" / \"algo contrario\" / \"algo para levantarme\""
            )
            save_conversation_history(user_id, text, response_text, parsed)
            return response_text

        # Si ya tenemos estrategia desde el primer mensaje
        parsed["match_strategy"] = strategy
        response_text = self._build_recommendation_text(parsed)
        save_conversation_history(user_id, text, response_text, parsed)
        return response_text
