"""Simple beer recommendations backed by a local SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

DB_PATH = Path(__file__).resolve().parent / "data" / "beers.db"

PAIRING_KEYWORDS: dict[str, tuple[str, ...]] = {
    "стейк": ("стейк", "мясо", "гриль", "говядина"),
    "сыры": ("сыр", "сыра", "сырная тарелка"),
    "десерт": ("десерт", "шоколад", "торт", "кейк", "брауни"),
    "острое": ("острое", "спайси", "чили", "азиатская"),
    "устрицы": ("устрица", "морепродукт", "рыба", "суши"),
    "вечеринка": ("вечеринка", "party", "друзья", "вечер"),
    "подарок": ("подарок", "gift", "сувенир"),
    "пикник": ("пикник", "на природе", "outdoor"),
}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_seed_data(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS beers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            style TEXT NOT NULL,
            flavor_profile TEXT NOT NULL,
            abv REAL NOT NULL,
            origin TEXT NOT NULL,
            description TEXT NOT NULL,
            facts TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS beer_pairings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context_label TEXT NOT NULL,
            category TEXT NOT NULL,
            style TEXT NOT NULL,
            beer_name TEXT NOT NULL,
            notes TEXT NOT NULL,
            fact TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pairing_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT NOT NULL,
            pairing_id INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pairing_id) REFERENCES beer_pairings(id) ON DELETE CASCADE
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER NOT NULL UNIQUE,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            onboarding_completed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users (telegram_id)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pairing_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pairing_id INTEGER NOT NULL,
            user_telegram_id INTEGER,
            message_id INTEGER,
            vote INTEGER NOT NULL, -- +1 like, -1 dislike
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pairing_id) REFERENCES beer_pairings(id) ON DELETE CASCADE
        )
        """
    )

    row = conn.execute("SELECT COUNT(*) FROM beers").fetchone()
    if row is None or row[0] > 0:
        _ensure_pairings_seed_data(conn)
        return

    beers: list[tuple[str, str, str, float, str, str, str]] = [
        (
            "Weihenstephaner Vitus",
            "Weizenbock",
            "сладкое, банан, гвоздика, мягкое",
            7.7,
            "Германия",
            "Пшеничный бок с богатыми фруктовыми и пряными нотами.",
            "Пивоварня основана в 1040 году; Vitus идеально подходит к сырам.",
        ),
        (
            "BrewDog Punk IPA",
            "American IPA",
            "горькое, цитрус, тропические фрукты, средняя крепость",
            5.6,
            "Шотландия",
            "Цитрусовая горечь и свежесть тропических фруктов.",
            "Пионер крафтового движения в Европе; название вдохновлено панк-культурой.",
        ),
        (
            "La Trappe Quadrupel",
            "Belgian Quadrupel",
            "сладкое, карамель, сухофрукты, согревающее",
            10.0,
            "Нидерланды",
            "Траппист с глубоким вкусом сухофруктов и специй.",
            "Один из семи официальных траппистских монастырей; Quadrupel выпускают с 1991 года.",
        ),
        (
            "Guinness Draught",
            "Dry Stout",
            "мягкое, кофейное, шоколад, низкая газированность",
            4.2,
            "Ирландия",
            "Классический стаут с кремовой плотной пеной.",
            "Подаётся через азотную смесь; рецептуру закрепили в 1959 году.",
        ),
        (
            "Saison Dupont",
            "Saison",
            "пряное, перец, сухое, освежающее",
            6.5,
            "Бельгия",
            "Фермерский сезон с сухим послевкусием и лёгкой перчинкой.",
            "Бутылки дображивают; Dupont варит saisons с 1844 года.",
        ),
        (
            "Baltika 6 Porter",
            "Baltic Porter",
            "темное, карамель, лакрица, цельный вкус",
            7.0,
            "Россия",
            "Насыщенный портер с нотами карамели и сушёных фруктов.",
            "Стиль возник на Балтике в XVIII веке; хорошо сочетается с десертами.",
        ),
        (
            "Põhjala Öö",
            "Imperial Baltic Porter",
            "сладкое, шоколад, ваниль, баррель",
            10.5,
            "Эстония",
            "Плотный портер, выдержанный на дубовой щепе.",
            "Põhjala — ведущая эстонская крафтовая пивоварня; название означает 'ночь'.",
        ),
        (
            "Hoegaarden Witbier",
            "Belgian Witbier",
            "светлое, цитрус, кориандр, освежающее",
            4.9,
            "Бельгия",
            "Классический бельгийский пшеничный эль с апельсиновой цедрой.",
            "Рецепт спас пивовар Пьер Селис; подают с долькой апельсина.",
        ),
    ]

    conn.executemany(
        """
        INSERT INTO beers (name, style, flavor_profile, abv, origin, description, facts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        beers,
    )
    conn.commit()
    _ensure_pairings_seed_data(conn)


def _ensure_pairings_seed_data(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT COUNT(*) FROM beer_pairings").fetchone()
    if row is not None and row[0] > 0:
        return

    pairings: list[tuple[str, str, str, str, str, str]] = [
        (
            "стейк на гриле",
            "food",
            "Imperial Stout",
            "Founders KBS",
            "Жареные ноты и карамелизированная корочка подчёркиваются плотным солодовым профилем стаута.",
            "KBS выдерживают на бочковой стружке, поэтому его часто советуют к красному мясу.",
        ),
        (
            "сырная тарелка",
            "food",
            "Saison",
            "Saison Dupont",
            "Пряная сухость баланcирует жирность мягких и полутвёрдых сыров.",
            "Пиво продолжает ферментировать в бутылке и раскрывается при подаче около 12°C.",
        ),
        (
            "десерт с шоколадом",
            "food",
            "Milk Stout",
            "Left Hand Milk Stout",
            "Лактоза и шоколадные ноты делают сочетание с брауни или фонданом почти десертом в десерте.",
            "Пиво варят с 1994 года и подают с нитро-разливом для шелковистой текстуры.",
        ),
        (
            "острые тако",
            "food",
            "American Pale Ale",
            "Sierra Nevada Pale Ale",
            "Цитрусовые хмели освежают и смягчают остроту сальсы и специи чили.",
            "Пиво считается иконой калифорнийского крафта и помогло популяризовать Cascade.",
        ),
        (
            "устрицы",
            "food",
            "Dry Stout",
            "Guinness Draught",
            "Солит сладковатый йодистый вкус моллюсков оттеняется мягкой кофейной горчинкой стаута.",
            "В XIX веке Guinness рекламировали как идеальное пиво к устрицам, традиция держится до сих пор.",
        ),
        (
            "вечеринка с друзьями",
            "occasion",
            "Session IPA",
            "BrewDog Hazy Jane",
            "Невысокая крепость и сочный тропический профиль позволяют пить пиво весь вечер.",
            "Вариант New England IPA с пониженным ABV; пиво нефильтрованное и очень ароматное.",
        ),
        (
            "подарок для гурмана",
            "occasion",
            "Belgian Quadrupel",
            "La Trappe Quadrupel",
            "Глубокий вкус сухофруктов и специй производит вау-эффект как у новичков, так и у коллекционеров.",
            "Настоящее траппистское пиво из Нидерландов; монастырь инвестирует прибыль в благотворительность.",
        ),
        (
            "пикник на природе",
            "occasion",
            "Belgian Witbier",
            "Hoegaarden Witbier",
            "Лёгкий цитрусовый профиль освежает и подходит как к салатам, так и к курице-гриль.",
            "Рецепт спасли от исчезновения в 1960-х; часто подают с апельсиновой долькой.",
        ),
    ]

    conn.executemany(
        """
        INSERT INTO beer_pairings (context_label, category, style, beer_name, notes, fact)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        pairings,
    )
    conn.commit()


def find_recommendations(preference_text: str, limit: int = 3) -> list[dict[str, str]]:
    terms = _normalize_terms(preference_text)
    query = "SELECT name, style, flavor_profile, abv, origin, description, facts FROM beers"
    params: list[str] = []

    if terms:
        like_clauses = ["(LOWER(style) LIKE ? OR LOWER(flavor_profile) LIKE ?)"] * len(terms)
        query += " WHERE " + " AND ".join(like_clauses)
        for term in terms:
            pattern = f"%{term}%"
            params.extend([pattern, pattern])

    query += " ORDER BY RANDOM() LIMIT ?"
    params.append(limit)

    conn = _connect()
    try:
        _ensure_seed_data(conn)
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    if rows:
        return [dict(row) for row in rows]

    # fallback: return random selection if filters were too strict
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT name, style, flavor_profile, abv, origin, description, facts "
            "FROM beers ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def find_pairings_for_context(query_text: str, limit: int = 1) -> list[dict[str, str]]:
    terms = _extract_pairing_terms(query_text)

    sql = (
        "SELECT id, context_label, category, style, beer_name, notes, fact "
        "FROM beer_pairings"
    )
    params: list[str | int] = []

    if terms:
        like_clauses = ["LOWER(context_label) LIKE ?"] * len(terms)
        sql += " WHERE " + " OR ".join(like_clauses)
        params.extend([f"%{term}%" for term in terms])

    sql += " ORDER BY RANDOM() LIMIT ?"
    params.append(limit)

    conn = _connect()
    try:
        _ensure_seed_data(conn)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def get_random_pairings(limit: int = 1) -> list[dict[str, str]]:
    conn = _connect()
    try:
        _ensure_seed_data(conn)
        rows = conn.execute(
            "SELECT id, context_label, category, style, beer_name, notes, fact "
            "FROM beer_pairings ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def save_pairing_history(user_query: str, pairing_ids: Iterable[int]) -> None:
    if not pairing_ids:
        return

    normalized_query = (user_query or "").strip()

    conn = _connect()
    try:
        _ensure_seed_data(conn)
        conn.executemany(
            "INSERT INTO pairing_history (user_query, pairing_id) VALUES (?, ?)",
            [(normalized_query, pairing_id) for pairing_id in pairing_ids],
        )
        conn.commit()
    finally:
        conn.close()


def save_pairing_feedback(
    pairing_id: int,
    vote: int,
    user_telegram_id: int | None = None,
    message_id: int | None = None,
) -> None:
    conn = _connect()
    try:
        _ensure_seed_data(conn)
        conn.execute(
            """
            INSERT INTO pairing_feedback (pairing_id, user_telegram_id, message_id, vote)
            VALUES (?, ?, ?, ?)
            """,
            (pairing_id, user_telegram_id, message_id, vote),
        )
        conn.commit()
    finally:
        conn.close()


def upsert_user(
    telegram_id: int,
    username: str | None,
    first_name: str | None,
    last_name: str | None,
) -> None:
    conn = _connect()
    try:
        _ensure_seed_data(conn)
        conn.execute(
            """
            INSERT INTO users (telegram_id, username, first_name, last_name)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(telegram_id) DO UPDATE SET
                username=excluded.username,
                first_name=excluded.first_name,
                last_name=excluded.last_name,
                updated_at=CURRENT_TIMESTAMP
            """,
            (telegram_id, username, first_name, last_name),
        )
        conn.commit()
    finally:
        conn.close()


def is_user_onboarded(telegram_id: int) -> bool:
    conn = _connect()
    try:
        _ensure_seed_data(conn)
        row = conn.execute(
            "SELECT onboarding_completed FROM users WHERE telegram_id = ?",
            (telegram_id,),
        ).fetchone()
    finally:
        conn.close()

    return bool(row and row["onboarding_completed"])


def mark_user_onboarded(telegram_id: int) -> None:
    conn = _connect()
    try:
        _ensure_seed_data(conn)
        conn.execute(
            """
            UPDATE users
            SET onboarding_completed = 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE telegram_id = ?
            """,
            (telegram_id,),
        )
        conn.commit()
    finally:
        conn.close()


def _normalize_terms(text: str) -> list[str]:
    if not text:
        return []

    tokens = [token.strip().lower() for token in text.split()]
    mapped: list[str] = []

    synonyms: dict[str, Iterable[str]] = {
        "слад": ["сладкое", "sweet"],
        "горч": ["горькое", "bitter", "ipa"],
        "темн": ["темное", "dark", "porter", "stout"],
        "фрукт": ["фруктовое", "fruit", "citrus"],
        "креп": ["крепкое", "strong", "quad"],
        "легк": ["легкое", "light", "wit"],
    }

    for token in tokens:
        mapped.append(token)
        for root, words in synonyms.items():
            if any(word in token for word in words):
                mapped.append(root)

    cleaned = [term for term in mapped if len(term) >= 3]
    # deduplicate while preserving order
    seen: set[str] = set()
    unique_terms: list[str] = []
    for term in cleaned:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    return unique_terms


def _extract_pairing_terms(text: str) -> list[str]:
    if not text:
        return []

    lowered = text.lower()
    matched: list[str] = []

    for keyword, synonyms in PAIRING_KEYWORDS.items():
        if keyword in lowered or any(token in lowered for token in synonyms):
            matched.append(keyword)

    if matched:
        return matched

    tokens = [token for token in lowered.split() if len(token) >= 3]
    return tokens[:3]


