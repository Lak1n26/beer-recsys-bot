-- Create table to store structured beer catalogue data sourced from JSON
CREATE TABLE IF NOT EXISTS beer_catalog (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    alcohol_percentage TEXT,
    bitterness TEXT,
    country TEXT,
    taste_tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    beer_type TEXT,
    style TEXT,
    raw JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS beer_catalog_country_idx
    ON beer_catalog (country);

CREATE INDEX IF NOT EXISTS beer_catalog_style_idx
    ON beer_catalog (style);

CREATE INDEX IF NOT EXISTS beer_catalog_taste_tags_gin_idx
    ON beer_catalog USING gin (taste_tags);

-- Load data from the JSON file located in the mounted data directory
WITH raw_doc AS (
    SELECT pg_read_file(
        '/beer-data/beer_data_full.json',
        0,
        10485760,
        false
    )::jsonb AS payload
), parsed AS (
    SELECT jsonb_array_elements(payload) AS item
    FROM raw_doc
), prepared AS (
    SELECT
        item->>'name' AS name,
        item->>'description' AS description,
        NULLIF(item->>'alcohol_percentage', '') AS alcohol_percentage,
        item->>'bitterness' AS bitterness,
        item->>'country' AS country,
        CASE
            WHEN item ? 'taste_tags'
                 AND jsonb_typeof(item->'taste_tags') = 'array'
            THEN ARRAY(
                SELECT jsonb_array_elements_text(item->'taste_tags')
            )
            ELSE ARRAY[]::TEXT[]
        END AS taste_tags,
        item->>'beer_type' AS beer_type,
        item->>'style' AS style,
        item AS raw
    FROM parsed
    WHERE NOT (item ? 'error')
      AND COALESCE(item->>'name', '') <> ''
      AND COALESCE(item->>'description', '') <> ''
)
INSERT INTO beer_catalog (
    name,
    description,
    alcohol_percentage,
    bitterness,
    country,
    taste_tags,
    beer_type,
    style,
    raw
)
SELECT
    name,
    description,
    alcohol_percentage,
    bitterness,
    country,
    taste_tags,
    beer_type,
    style,
    raw
FROM prepared
ON CONFLICT (name) DO UPDATE SET
    description = EXCLUDED.description,
    alcohol_percentage = EXCLUDED.alcohol_percentage,
    bitterness = EXCLUDED.bitterness,
    country = EXCLUDED.country,
    taste_tags = EXCLUDED.taste_tags,
    beer_type = EXCLUDED.beer_type,
    style = EXCLUDED.style,
    raw = EXCLUDED.raw,
    updated_at = NOW();

