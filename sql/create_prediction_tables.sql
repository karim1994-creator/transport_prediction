CREATE SCHEMA IF NOT EXISTS prediction;

CREATE TABLE IF NOT EXISTS prediction.frequentation_journaliere_ferre (
    prediction_id BIGSERIAL PRIMARY KEY,
    jour DATE NOT NULL,
    code_arret TEXT NOT NULL,
    id_zdc TEXT NOT NULL,
    mois SMALLINT,
    trimestre SMALLINT,
    semaine SMALLINT,
    jour_semaine SMALLINT,
    jour_du_mois SMALLINT,
    jour_annee SMALLINT,
    est_debut_mois SMALLINT,
    est_fin_mois SMALLINT,
    temperature_moyenne DOUBLE PRECISION,
    pluie_totale DOUBLE PRECISION,
    vitesse_vent_moyenne DOUBLE PRECISION,
    cat_jour TEXT,
    code_meteo TEXT,
    moyenne_historique_causale DOUBLE PRECISION,
    mediane_historique_causale DOUBLE PRECISION,
    moyenne_historique_cat_causale DOUBLE PRECISION,
    mediane_historique_cat_causale DOUBLE PRECISION,
    lag_1 DOUBLE PRECISION,
    lag_7 DOUBLE PRECISION,
    lag_14 DOUBLE PRECISION,
    moyenne_mobile_7 DOUBLE PRECISION,
    moyenne_mobile_28 DOUBLE PRECISION,
    nb_vald_predit DOUBLE PRECISION NOT NULL,
    nb_vald_reel DOUBLE PRECISION,
    modele TEXT NOT NULL,
    version_modele TEXT,
    date_heure_creation_ligne TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_ferre_jour
    ON prediction.frequentation_journaliere_ferre (jour DESC);
CREATE INDEX IF NOT EXISTS idx_pred_ferre_cle
    ON prediction.frequentation_journaliere_ferre (code_arret, id_zdc, jour DESC);

CREATE TABLE IF NOT EXISTS prediction.frequentation_journaliere_surface (
    prediction_id BIGSERIAL PRIMARY KEY,
    jour DATE NOT NULL,
    code_ligne TEXT NOT NULL,
    id_groupoflines TEXT NOT NULL,
    mois SMALLINT,
    trimestre SMALLINT,
    semaine SMALLINT,
    jour_semaine SMALLINT,
    jour_du_mois SMALLINT,
    jour_annee SMALLINT,
    est_debut_mois SMALLINT,
    est_fin_mois SMALLINT,
    temperature_moyenne DOUBLE PRECISION,
    pluie_totale DOUBLE PRECISION,
    vitesse_vent_moyenne DOUBLE PRECISION,
    cat_jour TEXT,
    code_meteo TEXT,
    moyenne_historique_causale DOUBLE PRECISION,
    mediane_historique_causale DOUBLE PRECISION,
    moyenne_historique_cat_causale DOUBLE PRECISION,
    mediane_historique_cat_causale DOUBLE PRECISION,
    lag_1 DOUBLE PRECISION,
    lag_7 DOUBLE PRECISION,
    lag_14 DOUBLE PRECISION,
    moyenne_mobile_7 DOUBLE PRECISION,
    moyenne_mobile_28 DOUBLE PRECISION,
    nb_vald_predit DOUBLE PRECISION NOT NULL,
    nb_vald_reel DOUBLE PRECISION,
    modele TEXT NOT NULL,
    version_modele TEXT,
    date_heure_creation_ligne TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_surface_jour
    ON prediction.frequentation_journaliere_surface (jour DESC);
CREATE INDEX IF NOT EXISTS idx_pred_surface_cle
    ON prediction.frequentation_journaliere_surface (code_ligne, id_groupoflines, jour DESC);

CREATE TABLE IF NOT EXISTS prediction.profils_horaires_ferre (
    prediction_id BIGSERIAL PRIMARY KEY,
    jour DATE NOT NULL,
    code_arret TEXT NOT NULL,
    id_zdc TEXT NOT NULL,
    cat_jour TEXT,
    heure_debut SMALLINT NOT NULL CHECK (heure_debut BETWEEN 0 AND 23),
    pourc_validations_predit DOUBLE PRECISION NOT NULL,
    pourc_validations_reel DOUBLE PRECISION,
    modele TEXT NOT NULL,
    version_modele TEXT,
    date_heure_creation_ligne TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_profil_ferre_jour
    ON prediction.profils_horaires_ferre (jour DESC);
CREATE INDEX IF NOT EXISTS idx_profil_ferre_cle
    ON prediction.profils_horaires_ferre (code_arret, id_zdc, cat_jour, jour DESC, heure_debut);

CREATE TABLE IF NOT EXISTS prediction.profils_horaires_surface (
    prediction_id BIGSERIAL PRIMARY KEY,
    jour DATE NOT NULL,
    code_ligne TEXT NOT NULL,
    id_groupofligne TEXT NOT NULL,
    cat_jour TEXT,
    heure_debut SMALLINT NOT NULL CHECK (heure_debut BETWEEN 0 AND 23),
    pourc_validations_predit DOUBLE PRECISION NOT NULL,
    pourc_validations_reel DOUBLE PRECISION,
    modele TEXT NOT NULL,
    version_modele TEXT,
    date_heure_creation_ligne TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_profil_surface_jour
    ON prediction.profils_horaires_surface (jour DESC);
CREATE INDEX IF NOT EXISTS idx_profil_surface_cle
    ON prediction.profils_horaires_surface (code_ligne, id_groupofligne, cat_jour, jour DESC, heure_debut);
