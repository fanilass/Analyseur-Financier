import io
import re
import base64
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# PDF
import pdfplumber

# ML
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ---------------------------
# Helpers
# ---------------------------
def sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp extremely large integers to 64-bit range to avoid Arrow overflow.
    Also attempts to coerce object columns containing integers to numeric.
    """
    if df is None or df.empty:
        return df
    df_clean = df.copy()
    int64_max = np.iinfo(np.int64).max
    int64_min = np.iinfo(np.int64).min

    for col in df_clean.columns:
        # Try convert to numeric when possible (without breaking text columns)
        coerced = pd.to_numeric(df_clean[col], errors="ignore")
        if pd.api.types.is_integer_dtype(coerced) or pd.api.types.is_float_dtype(coerced):
            # Clip values into int64 range, then keep as float if originally float
            if pd.api.types.is_float_dtype(coerced):
                df_clean[col] = coerced.clip(lower=float(int64_min), upper=float(int64_max))
            else:
                df_clean[col] = coerced.clip(lower=int64_min, upper=int64_max).astype(np.int64)
        else:
            # If it's object but contains big ints, try best-effort numeric conversion and clip
            tmp = pd.to_numeric(df_clean[col], errors="coerce")
            if tmp.notna().any():
                # Mixed column; keep original for non-numeric, use clipped for numeric
                clipped = tmp.clip(lower=float(int64_min), upper=float(int64_max))
                df_clean[col] = np.where(tmp.notna(), clipped, df_clean[col])
    return df_clean
def _to_int(s: str) -> int | None:
    if s is None:
        return None
    try:
        return int(str(s).replace(" ", "").replace(",", "").strip())
    except Exception:
        return None


def _to_float(s: str) -> float | None:
    if s is None:
        return None
    try:
        return float(str(s).replace(" ", "").replace(",", ".").strip())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t:
                text += t + "\n"
    return text


@st.cache_data(show_spinner=False)
def extract_tables_from_pdf_bytes(pdf_bytes: bytes) -> List[pd.DataFrame]:
    tables: List[pd.DataFrame] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                try:
                    tables.append(pd.DataFrame(table))
                except Exception:
                    continue
    return tables


@st.cache_data(show_spinner=False)
def extract_market_data(text: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    pattern = r"([\w\s\(\)]+?)\s*[:\s]\s*([\d\s,\.]+)"
    matches = re.findall(pattern, text)
    for key, val in matches:
        key_low = key.strip().lower()
        num = _to_float(val)
        if num is None:
            continue
        if "valeur totale" in key_low:
            data["total_valeur"] = num
        elif "volume" in key_low and "total" in key_low:
            data["total_volume"] = num
        elif "titres transiges" in key_low:
            data["titres_transiges"] = num
        elif "titres en hausse" in key_low:
            data["titres_hausse"] = num
        elif "titres en baisse" in key_low:
            data["titres_baisse"] = num
        elif "capitalisation" in key_low and "actions" in key_low:
            data["capitalisation_actions"] = num
        elif "capitalisation obligations" in key_low:
            data["capitalisation_obligations"] = num
    return data


@st.cache_data(show_spinner=False)
def extract_companies_from_text(text: str) -> pd.DataFrame:
    data: List[Dict] = []
    lines = text.split("\n")
    # Pattern inspired by notebook step 16
    pat = re.compile(
        r"([A-Z]{2,3})\s+([A-Z0-9]{2,6})\s+(.*?)\s+([\d\s]+)\s+([\d\s]+)\s+([\d\s]+).*?([+-]?\d+,\d+)\s*%"
    )
    for line in lines:
        m = pat.match(line)
        if not m:
            continue
        secteur, symbole, titre, volume, valeur, cours, variation = m.groups()
        try:
            data.append(
                {
                    "secteur": secteur,
                    "symbole": symbole,
                    "nom": titre.strip(),
                    "volume": _to_int(volume) or 0,
                    "valeur": _to_int(valeur) or 0,
                    "cours_actuel": _to_float(cours) or 0.0,
                    "variation": _to_float(variation) or 0.0,
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(data)
    if df.empty:
        return df

    mapping_secteurs = {
        "AGR": "Agriculture",
        "CB": "Consommation",
        "CD": "Distribution",
        "ENE": "Energie",
        "FIN": "Finance",
        "TEL": "Telecom",
        "IND": "Industrie",
        "SPU": "Services Publics",
    }
    df["secteur"] = df["secteur"].map(mapping_secteurs).fillna(df["secteur"])    
    return df


def engineer_features(df_companies: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    df = df_companies.copy()

    # Safe defaults
    for col in ["min_52s", "max_52s", "cap"]:
        if col not in df.columns:
            df[col] = np.nan if col != "cap" else df.get("valeur", pd.Series([0]*len(df)))

    df["volatilite_52s"] = (
        (df["max_52s"] - df["min_52s"]) / df["min_52s"].replace(0, np.nan)
    ) * 100
    df["ecart_max_52s"] = (
        (df["max_52s"] - df["cours_actuel"]) / df["max_52s"].replace(0, np.nan)
    ) * 100
    df["ecart_min_52s"] = (
        (df["cours_actuel"] - df["min_52s"]) / df["min_52s"].replace(0, np.nan)
    ) * 100
    df["ratio_valeur_volume"] = df["valeur"] / df["volume"].replace(0, 1)
    df["log_cap"] = np.log(df["cap"].replace(0, 1))

    if "secteur" not in df.columns:
        df["secteur"] = "N/A"

    le = LabelEncoder()
    df["secteur_encoded"] = le.fit_transform(df["secteur"].astype(str))

    if "cours_precedent" not in df.columns:
        df["cours_precedent"] = df["cours_actuel"].shift(1).fillna(df["cours_actuel"]) 

    if "position_52s" not in df.columns:
        den = (df["max_52s"] - df["min_52s"]).replace(0, 1)
        df["position_52s"] = ((df["cours_actuel"] - df["min_52s"]) / den) * 100

    return df, le


def train_predict_variation(df_feat: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    features = [
        "cours_precedent",
        "volume",
        "secteur_encoded",
        "volatilite_52s",
        "position_52s",
        "ecart_max_52s",
        "log_cap",
        "ratio_valeur_volume",
    ]

    X = df_feat[features].fillna(0)
    y = df_feat.get("variation", pd.Series([0.0] * len(df_feat)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    X_all_s = scaler.transform(X)
    var_pred = rf.predict(X_all_s)

    out = df_feat.copy()
    out["variation_pred"] = var_pred
    out["cours_pred"] = out["cours_actuel"] * (1 + out["variation_pred"] / 100)
    out["recommandation"] = out["variation_pred"].apply(lambda x: "ACHAT" if x > 2 else ("VENTE" if x < -2 else "NEUTRE"))

    metrics = {"mse": float(mse), "r2": float(r2), "rmse": float(np.sqrt(mse))}
    return out, metrics


def compute_risk(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    if "variation" not in df.columns:
        df["variation"] = df.get("variation_pred", pd.Series([0.0] * len(df)))

    df["score_risque"] = (
        (df.get("volatilite_52s", 0).fillna(0) * 0.4)
        + (df["variation"].abs() * 0.3)
        + ((100 - df.get("position_52s", 0).fillna(0)) * 0.3)
    ) / 3

    df["niveau_risque"] = pd.cut(
        df["score_risque"], bins=[0, 20, 40, 60, 100], labels=["FAIBLE", "MODÉRÉ", "ÉLEVÉ", "TRÈS ÉLEVÉ"]
    )

    return df


def build_strategies(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    strategies: Dict[str, pd.DataFrame] = {}

    # Croissance
    croissance = df[(df["variation_pred"] > 0) & (df["niveau_risque"].isin(["FAIBLE", "MODÉRÉ"]))].copy()
    strategies["Croissance"] = croissance.sort_values("variation_pred", ascending=False).head(5)

    # Valeur
    valeur = df[(df.get("position_52s", 100) < 50) & (df["variation_pred"] > -2)].copy()
    strategies["Valeur"] = valeur.sort_values("ecart_max_52s", ascending=False).head(5)

    # Dividendes (proxy: large cap + faible risque)
    dividendes = df[(df["log_cap"] > df["log_cap"].median()) & (df["niveau_risque"] == "FAIBLE")].copy()
    strategies["Dividendes"] = dividendes.sort_values("cap", ascending=False).head(5)

    # Spéculative
    speculative = df[(df["variation_pred"] > 3) | (df.get("volatilite_52s", 0) > 50)].copy()
    strategies["Spéculative"] = speculative.sort_values("variation_pred", ascending=False).head(3)

    return strategies


def make_summary_report(market: Dict[str, float], pred_df: pd.DataFrame, r2_value: float | None) -> str:
    lignes: List[str] = []
    lignes.append("TENDANCES GÉNÉRALES DU MARCHÉ")
    lignes.append(f"• Volume total échangé: {int(market.get('total_volume', 0)) if pd.notna(market.get('total_volume', np.nan)) else 'N/A'}")
    lignes.append(f"• Valeur totale: {int(market.get('total_valeur', 0)) if pd.notna(market.get('total_valeur', np.nan)) else 'N/A'} FCFA")
    lignes.append(
        f"• Titres en hausse: {int(market.get('titres_hausse', 0)) if pd.notna(market.get('titres_hausse', np.nan)) else 'N/A'} | Titres en baisse: {int(market.get('titres_baisse', 0)) if pd.notna(market.get('titres_baisse', np.nan)) else 'N/A'}"
    )

    lignes.append("")
    lignes.append("MEILLEURES PERFORMANCES")
    if "variation" in pred_df.columns and not pred_df["variation"].isna().all():
        for _, row in pred_df.nlargest(3, "variation").iterrows():
            lignes.append(f"• {row.get('nom', row.get('symbole', 'N/A'))} ({row.get('symbole', 'N/A')}): {row.get('variation', 0):+.2f}%")

    lignes.append("")
    lignes.append("PIRES PERFORMANCES")
    if "variation" in pred_df.columns and not pred_df["variation"].isna().all():
        for _, row in pred_df.nsmallest(3, "variation").iterrows():
            lignes.append(f"• {row.get('nom', row.get('symbole', 'N/A'))} ({row.get('symbole', 'N/A')}): {row.get('variation', 0):+.2f}%")

    lignes.append("")
    lignes.append("PRÉDICTIONS ALGORITHMIQUES")
    lignes.append(f"• Sociétés recommandées à l'ACHAT: {int((pred_df['recommandation']=='ACHAT').sum())}")
    lignes.append(f"• Sociétés recommandées à la VENTE: {int((pred_df['recommandation']=='VENTE').sum())}")
    lignes.append(f"• Sociétés NEUTRES: {int((pred_df['recommandation']=='NEUTRE').sum())}")

    lignes.append("")
    lignes.append("ANALYSE DE RISQUE (compte)" )
    for lvl in ["FAIBLE", "MODÉRÉ", "ÉLEVÉ", "TRÈS ÉLEVÉ"]:
        lignes.append(f"• Risque {lvl}: {int((pred_df['niveau_risque']==lvl).sum())} sociétés")

    lignes.append("")
    lignes.append("TOP 3 RECOMMANDATIONS ALGORITHMIQUES")
    top_reco = pred_df[pred_df["recommandation"] == "ACHAT"].head(3)
    for i, (_, row) in enumerate(top_reco.iterrows(), 1):
        lignes.append(f"{i}. {row.get('nom', row.get('symbole','N/A'))} ({row.get('symbole','N/A')})")
        lignes.append(f"   Prix actuel: {row.get('cours_actuel', 0):,.0f} FCFA")
        lignes.append(f"   Prédiction: {row.get('variation_pred', 0):+.2f}% | Risque: {row.get('niveau_risque', 'N/A')}")

    lignes.append("")
    if r2_value is not None:
        lignes.append("AVERTISSEMENTS")
        lignes.append(f"• Ces prédictions sont basées sur un modèle (R² = {r2_value:.3f})")
        lignes.append("• Les performances passées ne garantissent pas les résultats futurs")
        lignes.append("• Consultez un conseiller financier avant tout investissement")
        lignes.append("• Diversifiez toujours vos investissements")

    return "\n".join(lignes)


def qa_answer(question: str, text: str, df: pd.DataFrame, market: Dict[str, float]) -> str:
    q = question.lower().strip()
    if any(k in q for k in ["volume", "échang", "transig"]):
        v = market.get("total_volume")
        return f"Volume total échangé: {int(v) if v is not None else 'N/A'}"
    if any(k in q for k in ["valeur totale", "chiffre d'affaires", "valeur échang"]):
        v = market.get("total_valeur")
        return f"Valeur totale échangée: {int(v) if v is not None else 'N/A'} FCFA"
    if "capitalisation" in q:
        v = market.get("capitalisation_actions")
        return f"Capitalisation (Actions): {int(v) if v is not None else 'N/A'}"
    if q.startswith("qui") or "meilleure" in q or "top" in q:
        if not df.empty and "variation_pred" in df.columns:
            row = df.sort_values("variation_pred", ascending=False).head(1)
            if not row.empty:
                r = row.iloc[0]
                return f"Top prédiction: {r.get('nom', r.get('symbole','N/A'))} ({r.get('symbole','N/A')}) à {r.get('variation_pred',0):+.2f}%"
    # fallback: keyword-in-text search window
    words = q.split()
    for w in words:
        idx = text.lower().find(w)
        if idx != -1:
            start = max(0, idx - 120)
            end = min(len(text), idx + 200)
            return text[start:end].replace("\n", " ")
    return "Je n'ai pas trouvé d'information précise. Reformulez ou soyez plus spécifique."


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Analyse & Prédictions BRVM", page_icon="📈", layout="wide")

st.title("Analyse & Prédictions BRVM")
st.caption("Importer un bulletin PDF de la BRVM pour obtenir un résumé, des statistiques, des prédictions, une analyse de risque, des stratégies d’investissement et des visualisations.")

with st.sidebar:
    st.header("Importer le PDF")
    uploaded = st.file_uploader("Déposez le bulletin hebdomadaire (PDF)", type=["pdf"])
    st.markdown("---")
    st.subheader("Options")
    run_btn = st.button("Lancer l'analyse", type="primary", use_container_width=True)

if uploaded and run_btn:
    pdf_bytes = uploaded.read()
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)
    tables = extract_tables_from_pdf_bytes(pdf_bytes)
    market = extract_market_data(raw_text)
    df_companies = extract_companies_from_text(raw_text)

    # If companies not detected, try to fallback using tables (best-effort)
    if df_companies.empty and tables:
        # Heuristic: concatenate tables and attempt simple parsing
        try:
            merged = pd.concat(tables, ignore_index=True)
            merged.columns = [str(c) for c in merged.columns]
            # Attempt to find columns that look like expected fields
            guess_cols = [
                ("symbole", ["symbole", "ticker", 1]),
                ("nom", ["nom", "titre", 2]),
                ("cours_actuel", ["cours", "dernier", 5]),
                ("variation", ["variation", "%", 6]),
                ("volume", ["volume", 3]),
                ("valeur", ["valeur", 4]),
            ]
            parsed = {}
            for k, hints in guess_cols:
                col = None
                for h in hints:
                    if isinstance(h, int) and h < merged.shape[1]:
                        col = merged.columns[h]
                        break
                    matches = [c for c in merged.columns if str(c).strip().lower().find(str(h)) != -1]
                    if matches:
                        col = matches[0]
                        break
                if col is not None:
                    parsed[k] = merged[col]
            df_companies = pd.DataFrame(parsed)
            for c in ["volume", "valeur"]:
                if c in df_companies.columns:
                    df_companies[c] = df_companies[c].apply(_to_int).fillna(0)
            if "cours_actuel" in df_companies.columns:
                df_companies["cours_actuel"] = df_companies["cours_actuel"].apply(_to_float).fillna(0)
            if "variation" in df_companies.columns:
                df_companies["variation"] = df_companies["variation"].apply(_to_float).fillna(0)
        except Exception:
            pass

    # Engineer features, train and predict
    df_feat, _ = engineer_features(df_companies)
    pred_df, metrics = train_predict_variation(df_feat)
    risk_df = compute_risk(pred_df)
    strategies = build_strategies(risk_df)

    st.session_state["raw_text"] = raw_text
    st.session_state["market"] = market
    st.session_state["companies"] = df_companies
    st.session_state["pred"] = pred_df
    st.session_state["risk"] = risk_df
    st.session_state["strategies"] = strategies
    st.session_state["metrics"] = metrics


if "pred" in st.session_state:
    raw_text = st.session_state["raw_text"]
    market = st.session_state["market"]
    df_companies = st.session_state["companies"]
    pred_df = st.session_state["pred"]
    risk_df = st.session_state["risk"]
    strategies = st.session_state["strategies"]
    metrics = st.session_state["metrics"]

    tab_synth, tab_soc, tab_pred, tab_risk, tab_strat, tab_viz, tab_qa = st.tabs(
        [
            "Synthèse",
            "Sociétés",
            "Prédictions",
            "Risque",
            "Stratégies",
            "Visualisations",
            "Questions",
        ]
    )

    with tab_synth:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Volume total échangé", f"{int(market.get('total_volume', 0)) if pd.notna(market.get('total_volume', np.nan)) else 'N/A'}")
        col2.metric("Valeur totale (FCFA)", f"{int(market.get('total_valeur', 0)) if pd.notna(market.get('total_valeur', np.nan)) else 'N/A'}")
        col3.metric("Capitalisation Actions", f"{int(market.get('capitalisation_actions', 0)) if pd.notna(market.get('capitalisation_actions', np.nan)) else 'N/A'}")
        col4.metric("# Sociétés analysées", f"{len(df_companies) if isinstance(df_companies, pd.DataFrame) else 0}")

        st.markdown("---")
        st.subheader("Statistiques descriptives")
        num_cols = [c for c in ["cours_actuel", "variation", "volume", "valeur"] if c in df_companies.columns]
        if num_cols:
            st.dataframe(df_companies[num_cols].describe().round(2))
        else:
            st.info("Aucune donnée numérique disponible pour les statistiques.")

        st.markdown("---")
        st.subheader("Rapport de synthèse")
        report = make_summary_report(market, risk_df, metrics.get("r2"))
        st.text(report)

    with tab_soc:
        st.subheader("Sociétés (noms exacts)")
        display_cols = [c for c in ["symbole", "nom", "secteur", "cours_actuel", "variation", "volume", "valeur"] if c in df_companies.columns]
        df_disp = sanitize_df_for_streamlit(df_companies[display_cols].sort_values("nom" if "nom" in df_companies.columns else "symbole"))
        st.dataframe(df_disp)

    with tab_pred:
        st.subheader("Prédictions pour toutes les sociétés")
        cols = [c for c in ["symbole", "nom", "cours_actuel", "variation_pred", "cours_pred", "recommandation"] if c in pred_df.columns]
        st.dataframe(sanitize_df_for_streamlit(pred_df[cols].sort_values("variation_pred", ascending=False)))

        st.markdown("### Top 5 recommandations d'achat")
        st.dataframe(sanitize_df_for_streamlit(pred_df.sort_values("variation_pred", ascending=False).head(5)[cols]))

        st.markdown("### Top 5 recommandations de vente")
        st.dataframe(sanitize_df_for_streamlit(pred_df.sort_values("variation_pred", ascending=True).head(5)[cols]))

        st.caption(f"Performance modèle — R²: {metrics['r2']:.3f} | RMSE: {metrics['rmse']:.3f}")

    with tab_risk:
        st.subheader("Analyse de risque par société")
        cols = [c for c in ["symbole", "nom", "score_risque", "niveau_risque", "recommandation"] if c in risk_df.columns]
        st.dataframe(sanitize_df_for_streamlit(risk_df[cols].sort_values("score_risque")))

    with tab_strat:
        st.subheader("Stratégies d'investissement")
        for k, dfk in strategies.items():
            st.markdown(f"#### {k}")
            cols = [c for c in ["symbole", "nom", "cours_actuel", "variation_pred", "niveau_risque"] if c in dfk.columns]
            if not dfk.empty:
                st.dataframe(sanitize_df_for_streamlit(dfk[cols]))
            else:
                st.info("Aucune société ne correspond aux critères")

    with tab_viz:
        st.subheader("Visualisations")
        # Distribution variations réelles
        if "variation" in risk_df.columns and not risk_df["variation"].isna().all():
            fig = px.histogram(risk_df, x="variation", nbins=25, title="Distribution des variations (%)")
            st.plotly_chart(fig, use_container_width=True)

        # Performance par secteur
        if "secteur" in risk_df.columns and "variation" in risk_df.columns:
            perf = risk_df.groupby("secteur")["variation"].mean().reset_index().sort_values("variation")
            fig = px.bar(perf, x="variation", y="secteur", orientation="h", title="Performance moyenne par secteur")
            st.plotly_chart(fig, use_container_width=True)

        # Risk-return
        if "score_risque" in risk_df.columns and "variation_pred" in risk_df.columns:
            fig = px.scatter(
                risk_df,
                x="score_risque",
                y="variation_pred",
                color="cours_actuel" if "cours_actuel" in risk_df.columns else None,
                hover_data=[c for c in ["symbole", "nom"] if c in risk_df.columns],
                title="Analyse Risque-Rendement (prédite)",
                color_continuous_scale="Plasma",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Distribution recommandations
        if "recommandation" in risk_df.columns:
            rec = risk_df["recommandation"].value_counts().reset_index()
            rec.columns = ["Recommandation", "Count"]
            fig = px.pie(rec, names="Recommandation", values="Count", title="Distribution des recommandations")
            st.plotly_chart(fig, use_container_width=True)

    with tab_qa:
        st.subheader("Poser une question")
        q = st.text_input("Votre question")
        if st.button("Répondre"):
            if q.strip():
                st.write(qa_answer(q, raw_text, risk_df, market))
            else:
                st.info("Saisissez une question dans le champ ci-dessus.")

else:
    st.info("Importez un PDF et cliquez sur ‘Lancer l’analyse’. Un échantillon `boc.pdf` est présent dans le dossier pour tester localement.")


