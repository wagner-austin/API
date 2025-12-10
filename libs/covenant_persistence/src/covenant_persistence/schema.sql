-- Covenant Radar Database Schema
-- All monetary values stored as integers (cents or scaled by 1_000_000)

CREATE TABLE IF NOT EXISTS deals (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    borrower TEXT NOT NULL,
    sector TEXT NOT NULL,
    region TEXT NOT NULL,
    commitment_amount_cents BIGINT NOT NULL,
    currency TEXT NOT NULL,
    maturity_date DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS covenants (
    id UUID PRIMARY KEY,
    deal_id UUID NOT NULL REFERENCES deals(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    formula TEXT NOT NULL,
    threshold_value_scaled BIGINT NOT NULL,
    threshold_direction TEXT NOT NULL CHECK (threshold_direction IN ('<=', '>=')),
    frequency TEXT NOT NULL CHECK (frequency IN ('QUARTERLY', 'ANNUAL')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deal_id UUID NOT NULL REFERENCES deals(id) ON DELETE CASCADE,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value_scaled BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (deal_id, period_start, period_end, metric_name)
);

CREATE TABLE IF NOT EXISTS covenant_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    covenant_id UUID NOT NULL REFERENCES covenants(id) ON DELETE CASCADE,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    calculated_value_scaled BIGINT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('OK', 'NEAR_BREACH', 'BREACH')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (covenant_id, period_start, period_end)
);

CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    feature_names JSONB NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT FALSE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_covenants_deal_id ON covenants(deal_id);
CREATE INDEX IF NOT EXISTS idx_measurements_deal_period ON measurements(deal_id, period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_measurements_deal_id ON measurements(deal_id);
CREATE INDEX IF NOT EXISTS idx_results_covenant_id ON covenant_results(covenant_id);
CREATE INDEX IF NOT EXISTS idx_ml_models_active ON ml_models(is_active) WHERE is_active = TRUE;
