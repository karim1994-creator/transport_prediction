-- ============================================================
-- HISTORISATION DES PREDICTIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS prediction.ml_prediction_history (

    id BIGSERIAL PRIMARY KEY,

    prediction_timestamp TIMESTAMP NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    model_name VARCHAR(10) NOT NULL,

    model_version VARCHAR(100),

    perimeter VARCHAR(50),

    request_profile JSONB,

    prediction JSONB NOT NULL,

    prediction_total DOUBLE PRECISION,

    api_status VARCHAR(20) NOT NULL
        DEFAULT 'success',

    error_message TEXT
);


-- ============================================================
-- INDEX
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_ml_prediction_history_model
ON prediction.ml_prediction_history(model_name);

CREATE INDEX IF NOT EXISTS idx_ml_prediction_history_timestamp
ON prediction.ml_prediction_history(prediction_timestamp);


-- ============================================================
-- VERSION DES MODELES
-- ============================================================

CREATE TABLE IF NOT EXISTS prediction.ml_model_versions (

    id BIGSERIAL PRIMARY KEY,

    model_name VARCHAR(10) NOT NULL,

    version VARCHAR(100) NOT NULL,

    trained_at TIMESTAMP NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    artifact_path TEXT,

    evaluation_status VARCHAR(20),

    mae DOUBLE PRECISION,

    rmse DOUBLE PRECISION,

    r2 DOUBLE PRECISION,

    mlflow_run_id VARCHAR(100)
);


CREATE INDEX IF NOT EXISTS idx_ml_model_versions_model
ON prediction.ml_model_versions(model_name);

CREATE INDEX IF NOT EXISTS idx_ml_model_versions_date
ON prediction.ml_model_versions(trained_at);