CREATE TABLE IF NOT EXISTS public.ml_model_versions (

    id BIGSERIAL PRIMARY KEY,

    model_name VARCHAR(10) NOT NULL,

    version VARCHAR(100) NOT NULL,

    trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    artifact_path TEXT,

    evaluation_status VARCHAR(20),

    mae DOUBLE PRECISION,

    rmse DOUBLE PRECISION,

    r2 DOUBLE PRECISION,

    mlflow_run_id VARCHAR(100)

);