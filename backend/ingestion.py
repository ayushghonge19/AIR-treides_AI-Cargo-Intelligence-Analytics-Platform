import os
import re
from io import BytesIO, StringIO

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, MetaData, Table
from urllib.parse import unquote, quote_plus

load_dotenv()

REQUIRED_COLUMNS = [
    "month",
    "airport_name",
    "airline",
    "commodity_type",
    "export_import",
    "weight_tons",
]

# Fallback synonym mapping for robust column matching
COLUMN_MAPPING = {
    "month": ["month", "date", "year_month", "period", "timestamp", "datetime"],
    "airport_name": ["airport_name", "airport", "origin_airport", "destination_airport", "ap_name", "airportname"],
    "airline": ["airline", "carrier", "airline_name", "operator", "airlinename"],
    "commodity_type": ["commodity_type", "commodity", "cargo_type", "goods", "commoditytype"],
    "export_import": ["export_import", "exp_imp", "direction", "type", "flow", "exportimport"],
    "weight_tons": ["weight_tons", "weight", "tons", "weight_tonnes", "qty", "quantity", "weighttons"],
}


def get_safe_db_url(url: str) -> str:
    """URL-encode the database password if it has special characters to make it safe for SQLAlchemy."""
    if not url:
        return url
    pattern = r"^([^:]+://)([^:]+):(.*)@([^@]+)$"
    match = re.match(pattern, url)
    if match:
        scheme, username, password, rest = match.groups()
        safe_password = quote_plus(unquote(password))
        return f"{scheme}{username}:{safe_password}@{rest}"
    return url


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column headers to lowercase snake_case and map synonyms."""
    normalized = []
    for col in df.columns:
        name = str(col).strip().lower()
        name = re.sub(r"[\s/()]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        normalized.append(name)
    df.columns = normalized

    # Map synonyms to required names
    final_cols = {}
    for col in df.columns:
        mapped = False
        for target, synonyms in COLUMN_MAPPING.items():
            if col in synonyms:
                final_cols[col] = target
                mapped = True
                break
        if not mapped:
            final_cols[col] = col

    df = df.rename(columns=final_cols)
    return df


def _read_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel file into a DataFrame, handling custom quotes and delimiters."""
    raw = uploaded_file.getvalue()
    filename = getattr(uploaded_file, "name", "").lower()

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(raw))

    # Read CSV
    # Try decoding
    try:
        raw_text = raw.decode("utf-8-sig")
    except Exception:
        try:
            raw_text = raw.decode("utf-8")
        except Exception:
            raw_text = raw.decode("latin1")

    # Handle lines that are wrapped in outer double quotes (e.g., "col1,col2,col3")
    lines = raw_text.splitlines()
    cleaned_lines = []
    has_outer_quotes = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2:
            cleaned_lines.append(stripped[1:-1])
            has_outer_quotes = True
        else:
            cleaned_lines.append(line)

    if has_outer_quotes:
        raw_text = "\n".join(cleaned_lines)

    # Use sep=None, engine='python' to auto-detect delimiters (e.g. comma, semicolon, tab)
    return pd.read_csv(StringIO(raw_text), sep=None, engine='python')


def process_and_upload_file(uploaded_file, upsert: bool = False) -> dict:
    """
    Clean, validate, and persist air cargo records from CSV/Excel uploads.

    Returns:
        {"success": bool, "rows_inserted": int, "error": str}
    """
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        if not supabase_url:
            return {"success": False, "rows_inserted": 0, "error": "SUPABASE_URL not set."}

        safe_supabase_url = get_safe_db_url(supabase_url)
        engine = create_engine(safe_supabase_url)

        # Create upload_audit table if not exists
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS upload_audit (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    rows_inserted INTEGER NOT NULL,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.commit()

        # Read and normalize
        df = _read_file(uploaded_file)
        df = normalize_headers(df)

        # Validate required columns
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return {
                "success": False,
                "rows_inserted": 0,
                "error": f"Missing columns after normalization: {missing}. Columns found: {list(df.columns)}",
            }

        df = df[REQUIRED_COLUMNS].copy()

        # Type enforcement — dates
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        if df["month"].isna().any():
            return {
                "success": False,
                "rows_inserted": 0,
                "error": "Invalid month values. Expected parseable dates (e.g. YYYY-MM-DD).",
            }
        df["month"] = df["month"].dt.strftime("%Y-%m-%d")

        # Type enforcement — numeric weight
        df["weight_tons"] = pd.to_numeric(df["weight_tons"], errors="coerce")
        if df["weight_tons"].isna().any():
            return {
                "success": False,
                "rows_inserted": 0,
                "error": "Invalid weight_tons values. Expected numeric data.",
            }

        # Clean text columns
        for col in ["airport_name", "airline", "commodity_type", "export_import"]:
            df[col] = df[col].astype(str).str.strip()

        rows = len(df)
        if upsert:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            metadata = MetaData()
            table = Table("air_cargo_data", metadata, autoload_with=engine)
            records = df.to_dict(orient="records")
            
            chunk_size = 1000
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i+chunk_size]
                stmt = pg_insert(table).values(chunk)
                upsert_stmt = stmt.on_conflict_do_update(
                    index_elements=["month", "airport_name", "airline", "commodity_type", "export_import"],
                    set_=dict(weight_tons=stmt.excluded.weight_tons)
                )
                with engine.connect() as conn:
                    conn.execute(upsert_stmt)
                    conn.commit()
        else:
            df.to_sql(
                "air_cargo_data",
                engine,
                if_exists="append",
                index=False,
                method="multi",
            )

        # Log audit entry
        with engine.connect() as conn:
            conn.execute(
                text("INSERT INTO upload_audit (filename, rows_inserted) VALUES (:filename, :rows_inserted)"),
                {"filename": uploaded_file.name, "rows_inserted": rows}
            )
            conn.commit()

        return {"success": True, "rows_inserted": rows, "error": ""}

    except Exception as exc:
        return {"success": False, "rows_inserted": 0, "error": str(exc)}
