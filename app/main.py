"""Entrypoint for the Telegram beer bot service."""

from __future__ import annotations

import asyncio
import html
import logging
import random
from contextlib import suppress
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    Message,
)
from aiogram.exceptions import TelegramUnauthorizedError

from beer_search import search_similar_beers

from .beer_repo import (
    find_pairings_for_context,
    get_random_pairings,
    is_user_onboarded,
    mark_user_onboarded,
    save_pairing_feedback,
    save_pairing_history,
    upsert_user,
)
from .config import Settings, get_settings

PLACEHOLDER_REPLY = (
    "Пока не получилось найти точное совпадение. "
    "Попробуйте описать вкус, стиль или ситуацию чуть подробнее."
)

EMPTY_QUERY_REPLY = (
    "Напишите, какое пиво хотите: вкус, стиль, крепость или повод — "
    "и я подберу варианты."
)

PAIRING_REPLY_PROBABILITY = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEER_SEARCH_DIR = PROJECT_ROOT / "beer_search"
INDEX_PATH = BEER_SEARCH_DIR / "beer_faiss.index"
DATA_PATH = BEER_SEARCH_DIR / "beer_data_indexed.pkl"
MODEL_CACHE_DIR = BEER_SEARCH_DIR / "models"
RECOMMENDATION_TOP_K = 8
RECOMMENDATION_LIMIT = 3
MIN_RECOMMENDATION_SIMILARITY = 0.28
MAX_DESCRIPTION_LENGTH = 220


logger = logging.getLogger(__name__)


def _configure_logging(level_name: str) -> None:
    level_name = level_name.upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level)


async def _handle_start(message: Message) -> None:
    await _maybe_send_onboarding(message)
    await message.answer(
        "Готов подобрать пиво! Опишите вкус, стиль, крепость или повод — "
        "и я соберу подборку из каталога."
    )


async def _handle_help(message: Message) -> None:
    await _maybe_send_onboarding(message)
    await message.answer(
        "Напишите, какое пиво хотите. Я ищу похожие сорта по вкусам, "
        "стилям и описаниям, а команда /pairing помогает с food-пейрингом."
    )


async def _handle_pairing(message: Message) -> None:
    await _maybe_send_onboarding(message)
    query = _extract_pairing_query(message.text)
    pairing = _generate_pairing_block(query)
    if pairing:
        text, pairing_id = pairing
        markup = _build_feedback_keyboard(pairing_id)
        await message.answer(text, reply_markup=markup)
        return

    await message.answer(
        (
            "Пока не нашёл подходящую рекомендацию. Опишите блюдо или повод, "
            "например: <code>/pairing стейк на гриле</code>."
        )
    )


async def _handle_placeholder(message: Message) -> None:
    await _maybe_send_onboarding(message)

    user_text = (message.text or "").strip()
    if not user_text:
        await message.answer(EMPTY_QUERY_REPLY)
        return

    recommendations = await _fetch_vector_recommendations(user_text)
    if recommendations:
        reply_text = _format_recommendation_message(user_text, recommendations)
        await message.answer(reply_text, disable_web_page_preview=True)
    else:
        await message.answer(PLACEHOLDER_REPLY)

    # Optionally send a pairing as a separate message with buttons
    maybe = _maybe_generate_pairing(user_text)
    if maybe:
        text, pairing_id = maybe
        markup = _build_feedback_keyboard(pairing_id)
        await message.answer(text, reply_markup=markup)


def _maybe_generate_pairing(user_text: str) -> tuple[str, int] | None:
    if random.random() >= PAIRING_REPLY_PROBABILITY:
        return None

    return _generate_pairing_block(user_text)


def _generate_pairing_block(user_text: str) -> tuple[str, int] | None:
    pairings = find_pairings_for_context(user_text, limit=1)
    if not pairings:
        pairings = get_random_pairings(limit=1)
    if not pairings:
        return None

    pairing = pairings[0]
    save_pairing_history(user_text, [pairing["id"]])

    context_label = pairing["context_label"]
    style = pairing["style"]
    beer_name = pairing["beer_name"]
    notes = pairing["notes"]
    fact = pairing["fact"]

    text = (
        "🍽 <b>Режим «С чем пить»</b>\n"
        f"Контекст: {context_label}\n"
        f"Стиль: {style}\n"
        f"Бренд: {beer_name}\n"
        f"Почему подходит: {notes}\n"
        f"Факт: {fact}"
    )
    return text, int(pairing["id"])  # return text and pairing_id


def _extract_pairing_query(text: str | None) -> str:
    if not text:
        return ""

    stripped = text.strip()
    if not stripped.startswith("/pairing"):
        return stripped

    parts = stripped.split(maxsplit=1)
    if len(parts) == 1:
        return ""

    return parts[1].strip()


def _build_feedback_keyboard(pairing_id: int) -> InlineKeyboardMarkup:
    data_like = f"pairing_vote:{pairing_id}:up"
    data_dislike = f"pairing_vote:{pairing_id}:down"
    kb = [
        [
            InlineKeyboardButton(
                text="👍 Нравится",
                callback_data=data_like,
            ),
            InlineKeyboardButton(
                text="👎 Не подходит",
                callback_data=data_dislike,
            ),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=kb)


def _build_dispatcher() -> Dispatcher:
    router = Router()
    router.message.register(_handle_start, CommandStart())
    router.message.register(_handle_help, Command(commands=["help"]))
    router.message.register(
        _handle_pairing,
        Command(commands=["pairing"]),
    )
    router.message.register(_handle_placeholder)

    router.callback_query.register(
        _handle_vote,
        F.data.startswith("pairing_vote:"),
    )

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher


async def _configure_bot_commands(bot: Bot) -> None:
    commands = [
        BotCommand(command="start", description="Запуск и приветствие"),
        BotCommand(command="help", description="Как пользоваться ботом"),
        BotCommand(
            command="pairing",
            description="Подобрать пиво под блюдо или повод",
        ),
    ]
    await bot.set_my_commands(commands)


async def _start_polling(settings: Settings) -> None:
    token_secret = settings.telegram_token
    if token_secret is None:
        logger.error(
            "TELEGRAM_TOKEN is not set. Skipping polling startup."
        )
        return

    bot = Bot(
        token=token_secret.get_secret_value(),
        parse_mode=ParseMode.HTML,
    )
    dispatcher = _build_dispatcher()

    await _configure_bot_commands(bot)

    try:
        await dispatcher.start_polling(bot)
    except TelegramUnauthorizedError:
        logger.error(
            (
                "Telegram rejected the provided token. "
                "Please check TELEGRAM_TOKEN in .env."
            ),
        )


def run() -> None:
    """Run the Telegram bot that responds with a placeholder message."""

    settings = get_settings()
    _configure_logging(settings.log_level)
    if settings.telegram_token is None:
        logger.error(
            "TELEGRAM_TOKEN is not set. Bot will not start. "
            "Configure the token via environment or .env file."
        )
        return
    asyncio.run(_start_polling(settings))


async def _handle_vote(callback: CallbackQuery) -> None:
    data = callback.data or ""
    # Expected format: pairing_vote:<pairing_id>:<up|down>
    try:
        _, pid_str, vote_str = data.split(":", 2)
        pairing_id = int(pid_str)
        vote = 1 if vote_str == "up" else -1
    except Exception:
        await callback.answer("Некорректные данные", show_alert=False)
        return

    user_id = callback.from_user.id if callback.from_user else None
    message_id = callback.message.message_id if callback.message else None

    save_pairing_feedback(
        pairing_id=pairing_id,
        vote=vote,
        user_telegram_id=user_id,
        message_id=message_id,
    )

    # Acknowledge and disable buttons
    await callback.answer("Спасибо за отзыв!", show_alert=False)
    if callback.message:
        await callback.message.edit_reply_markup(reply_markup=None)


async def _fetch_vector_recommendations(query: str) -> list[dict[str, Any]]:
    try:
        return await asyncio.to_thread(
            search_similar_beers,
            query,
            top_k=RECOMMENDATION_TOP_K,
            min_similarity=MIN_RECOMMENDATION_SIMILARITY,
            show_full_description=False,
            verbose=False,
            model_cache_dir=MODEL_CACHE_DIR,
            index_path=INDEX_PATH,
            data_path=DATA_PATH,
        )
    except FileNotFoundError:
        logger.warning(
            "Vector search data is missing (index=%s, data=%s)",
            INDEX_PATH,
            DATA_PATH,
        )
    except Exception:
        logger.exception(
            "Failed to fetch vector recommendations for query '%s'",
            query,
        )
    return []


def _format_recommendation_message(
    query: str,
    beers: list[dict[str, Any]],
) -> str:
    header = (
        "🍺 <b>Подборка по запросу:</b> "
        f"<i>{html.escape(query)}</i>"
    )
    cards = [
        _format_recommendation_card(idx, beer)
        for idx, beer in enumerate(beers[:RECOMMENDATION_LIMIT], start=1)
    ]
    footer = (
        "🔁 Уточните вкус, стиль или повод — я подготовлю новую подборку. "
        "Команда /pairing подскажет сочетания с едой."
    )
    return "\n\n".join([header, *cards, footer]).strip()


def _format_recommendation_card(index: int, beer: dict[str, Any]) -> str:
    name = html.escape(str(beer.get("name", "Неизвестно")))
    style = html.escape(str(beer.get("style", "—")))
    abv = html.escape(str(beer.get("alcohol_percentage", "н/д")))
    country = html.escape(str(beer.get("country", "н/д")))
    similarity = beer.get("similarity_score")
    if isinstance(similarity, (int, float)):
        similarity_text = f"{similarity * 100:.0f}%"
    else:
        similarity_text = "—"

    tags = beer.get("taste_tags")
    if isinstance(tags, list) and tags:
        tag_text = ", ".join(html.escape(str(tag)) for tag in tags[:5])
    else:
        tag_text = ""

    description = _truncate_description(beer.get("description"))

    lines = [
        f"{index}. <b>{name}</b>",
        f"Стиль: {style} | ABV: {abv} | Страна: {country}",
    ]
    if similarity_text != "—":
        lines.append(f"Схожесть: {similarity_text}")
    if tag_text:
        lines.append(f"Теги: {tag_text}")
    if description:
        lines.append(f"Описание: {description}")

    return "\n".join(lines)


def _truncate_description(raw_description: Any) -> str:
    if not raw_description:
        return ""

    text = str(raw_description).replace("\n", " ").strip()
    if not text:
        return ""

    if len(text) > MAX_DESCRIPTION_LENGTH:
        text = text[:MAX_DESCRIPTION_LENGTH].rstrip() + "..."

    return html.escape(text)


async def _maybe_send_onboarding(message: Message) -> None:
    user = message.from_user
    if user is None:
        return

    upsert_user(
        telegram_id=user.id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
    )

    if is_user_onboarded(user.id):
        return

    preferred_name = user.first_name or user.username or "друг"
    safe_name = html.escape(preferred_name)

    onboarding_text = (
        f"👋 Привет, {safe_name}!\n\n"
        "<b>Как получить рекомендации:</b>\n"
        "• Опишите вкус, стиль, крепость или повод — я найду похожие сорта.\n"
        "• Используйте команду /pairing, чтобы подобрать пиво под блюдо "
        "или событие.\n"
        "• Оценивайте подборки кнопками — это помогает улучшать ответы.\n\n"
        "Попробуйте запрос: <i>легкое цитрусовое IPA для пикника</i>."
    )

    await message.answer(onboarding_text, disable_web_page_preview=True)
    mark_user_onboarded(user.id)


if __name__ == "__main__":  # pragma: no cover - manual run helper
    with suppress(KeyboardInterrupt):
        run()
