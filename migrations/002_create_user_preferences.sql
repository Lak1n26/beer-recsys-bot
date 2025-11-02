CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    beer_style_code INTEGER,
    flavor_profile_code INTEGER,
    bitterness_level SMALLINT,
    sweetness_level SMALLINT,
    sourness_level SMALLINT,
    maltiness_level SMALLINT,
    country_code INTEGER,
    occasion_code INTEGER,
    abv_preference NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS user_preferences_user_id_idx
    ON user_preferences (user_id);
-- Migration: create user_preferences table to store quantitative taste profiles

CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    beer_style_code SMALLINT,
    flavor_profile_code SMALLINT,
    bitterness_level SMALLINT,
    sweetness_level SMALLINT,
    sourness_level SMALLINT,
    maltiness_level SMALLINT,
    country_code SMALLINT,
    occasion_code SMALLINT,
    abv_preference SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_bitterness_range CHECK (bitterness_level BETWEEN 0 AND 10 OR bitterness_level IS NULL),
    CONSTRAINT chk_sweetness_range CHECK (sweetness_level BETWEEN 0 AND 10 OR sweetness_level IS NULL),
    CONSTRAINT chk_sourness_range CHECK (sourness_level BETWEEN 0 AND 10 OR sourness_level IS NULL),
    CONSTRAINT chk_maltiness_range CHECK (maltiness_level BETWEEN 0 AND 10 OR maltiness_level IS NULL),
    CONSTRAINT chk_abv_range CHECK (abv_preference BETWEEN 0 AND 20 OR abv_preference IS NULL),
    CONSTRAINT uq_user_preferences_user UNIQUE (user_id)
);

CREATE INDEX IF NOT EXISTS idx_user_preferences_style ON user_preferences (beer_style_code);
CREATE INDEX IF NOT EXISTS idx_user_preferences_country ON user_preferences (country_code);

