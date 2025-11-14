# app/chat.py
from typing import Dict, Any, List

from .utils import (
    parse_user_intent_with_openai,
    detect_strategy_from_text,
    infer_strategy_with_openai,
    save_conversation_history,
    get_movie_recommendations,
    get_series_recommendations,
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
        """
        Usa parsed[type, mood, match_strategy] para llamar a TMDB y
        construir el texto final de recomendación.
        - Para películas y series: usa TMDB.
        - Para música: por ahora solo texto placeholder (Spotify va después).
        """
        tipo = parsed.get("type")
        mood = parsed.get("mood")
        strategy = parsed.get("match_strategy")

        strategy_text = (
            "acompañar tu estado"
            if strategy == "match"
            else "cambiar tu estado / levantarte"
        )

        # PELÍCULAS
        if tipo == "movie":
            recs = get_movie_recommendations(parsed)
            if not recs:
                return (
                    f"Intenté buscar una **película** para mood **{mood}** y estrategia **{strategy_text}**, "
                    "pero no encontré resultados o hubo un problema con TMDB.\n"
                    "Verificá tu TMDB_API_KEY o probá describiéndome de otra forma qué querés ver."
                )

            lines: List[str] = [
                f"🎬 Te recomiendo estas películas para {strategy_text} estando **{mood}**:"
            ]
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
                    f"Intenté buscar una **serie** para mood **{mood}** y estrategia **{strategy_text}**, "
                    "pero no encontré resultados o hubo un problema con TMDB.\n"
                    "Verificá tu TMDB_API_KEY o probá describiéndome de otra forma qué querés ver."
                )

            lines: List[str] = [
                f"📺 Te recomiendo estas series para {strategy_text} estando **{mood}**:"
            ]
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

        # MÚSICA (placeholder hasta conectar Spotify)
        if tipo == "music":
            return (
                f"Te voy a recomendar **música** con mood **{mood}** y estrategia **{strategy_text}**, "
                "pero la integración con Spotify todavía no está lista. Ese será el próximo paso 🚧🎧"
            )

        # Tipo desconocido
        return (
            "Se me mezcló un poco el contexto 😅, probá pidiéndome de nuevo música, película o serie."
        )

    # ---------------------------
    # Lógica principal del bot
    # ---------------------------
    def handle_message(self, user_id: str, text: str) -> str:
        """
        Maneja un mensaje del usuario.
        - Si hay una intención pendiente, este mensaje se interpreta como
          respuesta a "¿acompañar o cambiar el ánimo?".
        - Si no, se interpreta como un nuevo pedido (music/movie/series + mood).
        """

        # 1) Si ya teníamos una intención pendiente, este mensaje es la ESTRATEGIA
        if user_id in self.pending_intents:
            parsed = self.pending_intents[user_id]
            tipo = parsed["type"]
            mood = parsed["mood"]

            # 1.a) Intentar con OpenAI interpretar la estrategia
            strategy = infer_strategy_with_openai(text, parsed)

            # 1.b) Si OpenAI falló (sin key o error), usamos fallback por keywords
            if strategy is None:
                strategy = detect_strategy_from_text(text)

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

            # Ya tenemos estrategia → limpiamos pendiente
            parsed["match_strategy"] = strategy
            del self.pending_intents[user_id]

            # Construimos texto final con TMDB (o placeholder para música)
            response_text = self._build_recommendation_text(parsed)
            save_conversation_history(user_id, text, response_text, parsed)
            return response_text

        # 2) Si NO hay intención pendiente, este mensaje es un NUEVO pedido
        parsed = parse_user_intent_with_openai(text)
        tipo = parsed["type"]
        mood = parsed["mood"]

        # Para el primer mensaje, SOLO usamos el detector por keywords
        # para ver si ya dejó clara la estrategia.
        strategy = detect_strategy_from_text(text)

        # 2.a) Si AÚN no sabemos la estrategia → preguntamos y guardamos el intent
        if strategy is None:
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

        # 3) Si ya tenemos estrategia desde el primer mensaje
        parsed["match_strategy"] = strategy
        response_text = self._build_recommendation_text(parsed)

        save_conversation_history(user_id, text, response_text, parsed)
        return response_text
