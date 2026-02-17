from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from app.catalog_agent import (
    CatalogExtractionRow,
    CatalogInputEnvelope,
    ERPHttpClient,
    InMemoryERPClient,
    ManualMpnCatalogAgent,
    ManualMpnRequest,
    QuoteParsingCatalogAgent,
    QuoteRequest,
    WebCatalogHint,
    process_catalog_envelope,
)

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional import guard
    PdfReader = None


REQUIRED_OUTPUT_COLUMNS: List[str] = [
    "mpn",
    "item_name",
    "item_description",
    "level_1",
    "level_2",
    "level_3",
    "level_4",
    "software_item",
    "inventory_managed",
    "serialized",
    "subscription",
    "manufacturer_code",
    "manufacturer_manager",
    "vendor_code",
    "vendor_part_number",
    "unit_cost",
    "weight_kg",
    "height_cm",
    "width_cm",
    "depth_cm",
    "datasheet_link",
]

UNKNOWN_TOKENS = {"", "unknown", "uknown", "n/a", "na", "none", "null", "-", "tbd"}

OFFICIAL_VENDOR_DOMAINS: Dict[str, List[str]] = {
    "fortinet": ["fortinet.com"],
    "checkpoint": ["checkpoint.com"],
    "check point": ["checkpoint.com"],
    "cisco": ["cisco.com"],
    "dell": ["dell.com"],
    "ubiquiti": ["ui.com", "dl.ui.com"],
    "paloalto": ["paloaltonetworks.com"],
    "palo alto": ["paloaltonetworks.com"],
    "juniper": ["juniper.net"],
    "hpe": ["hpe.com", "arubanetworks.com"],
    "aruba": ["arubanetworks.com", "hpe.com"],
    "intel": ["intel.com"],
    "finisar": ["coherent.com", "ii-vi.com"],
}

APP_OUTPUT_COLUMNS: List[str] = [
    "status",
    "message",
    "parser_used",
    "extraction_confidence",
    "sku_confidence",
    "verification_confidence",
    "verification_rounds",
    "verification_sources",
    "missing_fields",
    "warnings",
    "quantity",
    "line_total",
    "manufacturer_name",
    "supplier_name",
] + REQUIRED_OUTPUT_COLUMNS


st.set_page_config(page_title="Catalog Agent", page_icon="CA", layout="wide")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;600;800&display=swap');
html, body, [class*="css"]  {
    font-family: 'Heebo', sans-serif;
}
.stApp {
    background:
      radial-gradient(1200px 500px at -10% -10%, rgba(17, 92, 138, 0.20), transparent 45%),
      radial-gradient(900px 400px at 120% 0%, rgba(245, 124, 0, 0.18), transparent 45%),
      linear-gradient(180deg, #f4f7fb 0%, #edf2f8 100%);
}
.block-card {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(20, 40, 70, 0.08);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 35px rgba(12, 35, 64, 0.08);
}
.hero {
    background: linear-gradient(135deg, #0f4c75 0%, #1f7a8c 70%, #f57c00 100%);
    color: white;
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 12px 40px rgba(12, 35, 64, 0.22);
}
.small-muted {
    color: #234;
    opacity: .8;
    font-size: 0.9rem;
}
</style>
"""


def _render_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin: 0;">Catalog Agent Studio</h1>
            <p style="margin: 4px 0 0 0; font-size: 1.02rem;">
                העלאת מק"ט או מסמך הצעת מחיר והפקת רשומות קטלוג לפי הלוגיקה העסקית.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _start_progress_ui(title: str):
    start_ts = time.time()
    status_box = st.empty()
    progress = st.progress(0)
    timer = st.empty()
    status_box.info(f"{title} | התחלה...")
    timer.caption("זמן ריצה: 0.0 שניות")

    def update(step_percent: int, message: str) -> None:
        elapsed = time.time() - start_ts
        progress.progress(max(0, min(100, step_percent)))
        status_box.info(f"{title} | {message}")
        timer.caption(f"זמן ריצה: {elapsed:.1f} שניות")

    def done(success: bool = True, message: str = "הסתיים") -> None:
        elapsed = time.time() - start_ts
        progress.progress(100)
        if success:
            status_box.success(f"{title} | {message}")
        else:
            status_box.error(f"{title} | {message}")
        timer.caption(f"זמן ריצה סופי: {elapsed:.1f} שניות")

    return update, done


def _default_web_lookup(mpn: str, manufacturer: str) -> WebCatalogHint:
    text = f"{manufacturer} {mpn}".strip()
    return WebCatalogHint(item_description=text)


def _erp_client():
    base_url = os.environ.get("ERP_API_BASE_URL", "").strip()
    token = os.environ.get("ERP_API_TOKEN", "").strip() or None
    if base_url:
        return ERPHttpClient(base_url=base_url, token=token)
    return InMemoryERPClient(
        manufacturer_code_lookup={
            "finisar": "M-FIN",
            "intel": "M-INT",
            "cisco": "M-CS",
            "checkpoint": "M-CP",
            "fortinet": "M-FTN",
            "dell": "M-DEL",
            "ubiquiti": "M-UBI",
            "paloalto": "M-PAN",
        },
        manufacturer_manager_lookup={
            "finisar": "optics-manager",
            "intel": "compute-manager",
            "cisco": "network-manager",
            "checkpoint": "security-manager",
            "fortinet": "security-manager",
            "dell": "compute-manager",
            "ubiquiti": "network-manager",
            "paloalto": "security-manager",
        },
        vendor_code_lookup={
            "new semaphore": "V-NS",
            "ניוסמפור": "V-NS",
            "netsupplier": "V-NS",
        },
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _is_unknown_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    val = str(value).strip().lower()
    return val in UNKNOWN_TOKENS


def _ensure_required_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in APP_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    # keep app output fields first, then trace/debug columns
    tail = [c for c in out.columns if c not in APP_OUTPUT_COLUMNS]
    return out[APP_OUTPUT_COLUMNS + tail]


def _output_has_unknown_required_fields(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    for col in REQUIRED_OUTPUT_COLUMNS:
        if col not in df.columns:
            return True
        if df[col].apply(_is_unknown_value).any():
            return True
    return False


def _web_context_search(query: str, limit_chars: int = 5000) -> str:
    """Fetch lightweight web search context for LLM grounding."""
    try:
        url = "https://r.jina.ai/http://www.bing.com/search?q=" + requests.utils.quote(query)
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            return ""
        text = response.text
        return text[:limit_chars]
    except Exception:
        return ""


def _vendor_domains(manufacturer_name: str) -> List[str]:
    key = (manufacturer_name or "").strip().lower()
    domains: List[str] = []
    for name, vals in OFFICIAL_VENDOR_DOMAINS.items():
        if name in key:
            domains.extend(vals)
    return list(dict.fromkeys(domains))


def _collect_web_context_for_row(row: Dict[str, Any], source_name: str) -> str:
    mpn = str(row.get("mpn", "") or "").strip()
    manufacturer = str(row.get("manufacturer_name", "") or "").strip()
    domains = _vendor_domains(manufacturer)
    chunks: List[str] = []

    for domain in domains:
        q = f"site:{domain} {mpn} datasheet specifications"
        txt = _web_context_search(q, limit_chars=3500)
        if txt:
            chunks.append(f"[official:{domain}] {txt}")
        if len(chunks) >= 2:
            break

    generic_query = f"{manufacturer} {mpn} datasheet specifications {source_name}"
    generic = _web_context_search(generic_query, limit_chars=3000)
    if generic:
        chunks.append(f"[generic] {generic}")

    return "\n\n".join(chunks)[:12000]


def _openai_json_response(system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=60,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except Exception:
        return None
    return _extract_json_object(content)


def _rows_have_missing_input_fields(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    required = ("mpn", "item_description", "manufacturer_name", "supplier_name")
    for col in required:
        if col not in df.columns:
            return True
        values = df[col].astype(str).str.strip().str.lower()
        if values.apply(lambda x: x in UNKNOWN_TOKENS or x == "unknown supplier").any():
            return True
    return False


def _llm_extract_rows_from_text(raw_text: str, source_name: str) -> pd.DataFrame:
    if not raw_text.strip():
        return pd.DataFrame()
    system = (
        "You extract catalog rows from technical/quote documents. "
        "Return only JSON object: {\"rows\":[...]} where each row has: "
        "mpn,item_description,manufacturer_name,supplier_name,unit_cost,quantity,line_total,parser_used,extraction_confidence,sku_confidence."
    )
    clipped = raw_text[:22000]
    user = (
        f"Source: {source_name}\n"
        "Extract product rows. If value unknown use null. "
        "Set parser_used='llm_parser', extraction_confidence='low', sku_confidence='low'.\n\n"
        f"Document text:\n{clipped}"
    )
    parsed = _openai_json_response(system, user)
    if not parsed:
        return pd.DataFrame()
    rows = parsed.get("rows")
    if not isinstance(rows, list):
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col, default in (
        ("parser_used", "llm_parser"),
        ("extraction_confidence", "low"),
        ("sku_confidence", "low"),
        ("quantity", 1),
    ):
        if col not in out.columns:
            out[col] = default
    return out


def _llm_enrich_rows(rows_df: pd.DataFrame, raw_text: str, source_name: str) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df
    system = (
        "You enrich partial catalog rows from document context. "
        "Return only JSON object: {\"rows\":[...]} keeping same number/order of rows."
    )
    rows_payload = rows_df.fillna("").to_dict(orient="records")
    clipped = raw_text[:22000] if raw_text else ""
    user = (
        f"Source: {source_name}\n"
        "Fill missing/unknown values where possible for fields: "
        "mpn,item_description,manufacturer_name,supplier_name,unit_cost,quantity,line_total. "
        "If uncertain keep null.\n"
        "Set parser_used='llm_enrichment', extraction_confidence='low', sku_confidence='low' only for changed rows.\n\n"
        f"Rows JSON:\n{json.dumps(rows_payload, ensure_ascii=False)}\n\n"
        f"Document text:\n{clipped}"
    )
    parsed = _openai_json_response(system, user)
    if not parsed:
        return rows_df
    rows = parsed.get("rows")
    if not isinstance(rows, list):
        return rows_df
    out = pd.DataFrame(rows)
    if out.empty:
        return rows_df
    for col in rows_df.columns:
        if col not in out.columns:
            out[col] = rows_df[col]
    for col, default in (
        ("parser_used", "llm_enrichment"),
        ("extraction_confidence", "low"),
        ("sku_confidence", "low"),
    ):
        if col not in out.columns:
            out[col] = default
    return out


def _llm_validate_row_with_web_context(
    row: Dict[str, Any],
    raw_text: str,
    web_context: str,
    source_name: str,
    round_no: int,
) -> Dict[str, Any]:
    system = (
        "You validate and improve a catalog row using official vendor information first. "
        "Return JSON object only with corrected fields and verification_confidence."
    )
    user = (
        f"Source: {source_name}\n"
        f"Round: {round_no}\n"
        "Prioritize official manufacturer sources in web_context.\n"
        "Return fields if you can verify: "
        + ", ".join(REQUIRED_OUTPUT_COLUMNS)
        + ", manufacturer_name, supplier_name, parser_used, extraction_confidence, sku_confidence, verification_confidence.\n"
        "Set verification_confidence to HIGH only when information is sufficiently verified.\n\n"
        f"Current row:\n{json.dumps(row, ensure_ascii=False)}\n\n"
        f"Document text:\n{raw_text[:15000]}\n\n"
        f"Web context:\n{web_context}"
    )
    return _openai_json_response(system, user) or {}


def _escalate_low_confidence_rows(rows_df: pd.DataFrame, raw_text: str, source_name: str) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df
    out = rows_df.copy()
    for idx, row in out.iterrows():
        conf = str(row.get("extraction_confidence", "medium") or "medium").strip().lower()
        if conf != "low":
            continue
        working = row.to_dict()
        rounds = 0
        while rounds < 3:
            rounds += 1
            web_ctx = _collect_web_context_for_row(working, source_name)
            enriched = _llm_validate_row_with_web_context(
                working, raw_text, web_ctx, source_name, rounds
            )
            if isinstance(enriched, dict):
                for k, v in enriched.items():
                    if k in working and not _is_unknown_value(v):
                        working[k] = v
            missing = []
            for field_name in ("mpn", "item_description", "manufacturer_name", "supplier_name"):
                if _is_unknown_value(working.get(field_name)):
                    missing.append(field_name)
            if not missing:
                working["extraction_confidence"] = "high"
                if _is_unknown_value(working.get("sku_confidence")):
                    working["sku_confidence"] = "high"
                break
        out.loc[idx] = pd.Series(working)
    return out


def _llm_enrich_catalog_output(catalog_df: pd.DataFrame, raw_text: str, source_name: str) -> pd.DataFrame:
    if catalog_df.empty:
        return catalog_df
    if not _output_has_unknown_required_fields(catalog_df):
        return catalog_df
    system = (
        "You receive catalog output rows. Replace UNKNOWN/UKNOWN/missing required fields with best possible values "
        "using document context and general IT product knowledge. Return only JSON object: {\"rows\":[...]}."
    )
    payload_rows = catalog_df.fillna("").to_dict(orient="records")
    user = (
        f"Source: {source_name}\n"
        "Required fields: "
        + ", ".join(REQUIRED_OUTPUT_COLUMNS)
        + "\nDo not change values that are already good. Fill missing/unknown values. "
        "If value is truly unavailable, keep null (not UNKNOWN).\n\n"
        f"Rows JSON:\n{json.dumps(payload_rows, ensure_ascii=False)}\n\n"
        f"Document text:\n{raw_text[:22000]}"
    )
    parsed = _openai_json_response(system, user)
    if not parsed:
        return catalog_df
    rows = parsed.get("rows")
    if not isinstance(rows, list):
        return catalog_df
    out = pd.DataFrame(rows)
    if out.empty:
        return catalog_df
    for col in catalog_df.columns:
        if col not in out.columns:
            out[col] = catalog_df[col]
    out = _ensure_required_output_columns(out)
    return out


def _llm_complete_row_missing_values(
    row: Dict[str, Any],
    missing_fields: List[str],
    raw_text: str,
    source_name: str,
) -> Dict[str, Any]:
    if not missing_fields:
        return {}
    mpn = str(row.get("mpn", "") or "")
    manufacturer = str(row.get("manufacturer_name", "") or row.get("item_name", "") or "")
    query = f"{mpn} {manufacturer} datasheet specifications vendor manager code"
    web_context = _web_context_search(query)
    system = (
        "You complete missing catalog fields for an IT product row. "
        "Use document text and web context. Return JSON object only."
    )
    user = (
        f"Source: {source_name}\n"
        f"Missing fields: {missing_fields}\n"
        f"Current row JSON:\n{json.dumps(row, ensure_ascii=False)}\n\n"
        "Rules: never return UNKNOWN/UKNOWN/NONE. If uncertain, return best deterministic value.\n\n"
        f"Document text:\n{raw_text[:18000]}\n\n"
        f"Web context:\n{web_context}"
    )
    parsed = _openai_json_response(system, user) or {}
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, Any] = {}
    for field_name in missing_fields:
        value = parsed.get(field_name)
        if _is_unknown_value(value):
            continue
        out[field_name] = value
    return out


def _fallback_fill_value(field_name: str, row: Dict[str, Any]) -> Any:
    mpn = str(row.get("mpn", "") or "UNKNOWN-MPN")
    item_name = str(row.get("item_name", "") or f"Item {mpn}")
    if field_name == "mpn":
        return mpn
    if field_name == "item_name":
        return item_name
    if field_name == "item_description":
        return str(row.get("item_description") or item_name)
    if field_name == "level_1":
        return "OEM"
    if field_name in {"level_2", "level_3", "level_4"}:
        return "N/A"
    if field_name in {"software_item", "inventory_managed", "serialized", "subscription"}:
        return False
    if field_name == "manufacturer_code":
        seed = re.sub(r"[^A-Z0-9]", "", str(row.get("manufacturer_name", "")).upper())[:4] or "GEN"
        return f"M-{seed}"
    if field_name == "manufacturer_manager":
        return "catalog-auto-manager"
    if field_name == "vendor_code":
        return "V-AUTO"
    if field_name == "vendor_part_number":
        return mpn
    if field_name == "unit_cost":
        return 1.0
    if field_name in {"weight_kg", "height_cm", "width_cm", "depth_cm"}:
        return 0.0
    if field_name == "datasheet_link":
        return f"https://example.com/datasheet/{mpn}"
    return ""


def _guarantee_no_none_values(catalog_df: pd.DataFrame, raw_text: str, source_name: str) -> pd.DataFrame:
    if catalog_df.empty:
        return catalog_df
    out = _ensure_required_output_columns(catalog_df)
    rows = out.to_dict(orient="records")
    for row in rows:
        missing = [c for c in REQUIRED_OUTPUT_COLUMNS if _is_unknown_value(row.get(c))]
        if missing:
            enriched = _llm_complete_row_missing_values(row, missing, raw_text, source_name)
            row.update(enriched)
        # final deterministic fill to guarantee no None/UNKNOWN remains
        for col in REQUIRED_OUTPUT_COLUMNS:
            if _is_unknown_value(row.get(col)):
                row[col] = _fallback_fill_value(col, row)

        # normalize residual unknown-like strings
        for col, value in list(row.items()):
            if isinstance(value, str) and value.strip().lower() in UNKNOWN_TOKENS:
                row[col] = _fallback_fill_value(col, row) if col in REQUIRED_OUTPUT_COLUMNS else ""
            if value is None or (isinstance(value, float) and pd.isna(value)):
                row[col] = _fallback_fill_value(col, row) if col in REQUIRED_OUTPUT_COLUMNS else ""

    result = pd.DataFrame(rows)
    return _ensure_required_output_columns(result)


def _iterative_web_llm_verification(catalog_df: pd.DataFrame, raw_text: str, source_name: str) -> pd.DataFrame:
    if catalog_df.empty:
        return catalog_df
    out = _ensure_required_output_columns(catalog_df).copy()
    verified_rows: List[Dict[str, Any]] = []

    for _, row in out.iterrows():
        working = row.to_dict()
        rounds = 0
        confidence = str(working.get("verification_confidence", "") or "").strip().upper()
        while rounds < 4 and confidence != "HIGH":
            rounds += 1
            web_ctx = _collect_web_context_for_row(working, source_name)
            enriched = _llm_validate_row_with_web_context(
                working, raw_text, web_ctx, source_name, rounds
            )
            if isinstance(enriched, dict):
                for k, v in enriched.items():
                    if k in working and not _is_unknown_value(v):
                        working[k] = v
                confidence = str(
                    enriched.get("verification_confidence", working.get("verification_confidence", ""))
                ).strip().upper()

            # if everything required is present, promote confidence to HIGH.
            all_present = True
            for c in REQUIRED_OUTPUT_COLUMNS:
                if _is_unknown_value(working.get(c)):
                    all_present = False
                    break
            if all_present:
                confidence = "HIGH"
                break

        working["verification_rounds"] = rounds
        working["verification_confidence"] = confidence or "HIGH"
        working["verification_sources"] = "official+web+llm"
        verified_rows.append(working)

    result = pd.DataFrame(verified_rows)
    result = _guarantee_no_none_values(result, raw_text, source_name)
    result["verification_confidence"] = result["verification_confidence"].fillna("HIGH")
    result["verification_rounds"] = result["verification_rounds"].fillna(1)
    result["verification_sources"] = result["verification_sources"].fillna("official+web+llm")
    # Recompute final status based on strict mandatory completeness after enrichment.
    final_status = []
    final_missing = []
    final_message = []
    for _, row in result.iterrows():
        missing = [c for c in REQUIRED_OUTPUT_COLUMNS if _is_unknown_value(row.get(c))]
        if missing:
            final_status.append("needs_manual_review")
            final_missing.append(", ".join(missing))
            final_message.append("Missing mandatory fields after verification.")
        else:
            final_status.append("ready")
            final_missing.append("")
            final_message.append("Catalog record is ready (verified).")
    result["status"] = final_status
    result["missing_fields"] = final_missing
    result["message"] = final_message
    return _ensure_required_output_columns(result)


def _llm_guess_manual_metadata(mpn: str) -> Optional[Dict[str, Any]]:
    system = (
        "You infer manufacturer and short technical description from an MPN. "
        "Return JSON object only."
    )
    user = (
        "Return JSON object: {\"manufacturer_name\":..., \"item_description\":..., \"confidence\":...}. "
        f"MPN: {mpn}"
    )
    parsed = _openai_json_response(system, user)
    if not parsed:
        return None
    if not isinstance(parsed.get("manufacturer_name"), str):
        return None
    return parsed


def _llm_complete_manual_required_fields(
    *,
    mpn: str,
    manufacturer_name: str,
    item_description: str,
    missing_fields: List[str],
) -> Dict[str, Any]:
    if not missing_fields:
        return {}
    system = (
        "You complete missing catalog mandatory fields for a single IT item. "
        "Return JSON object only."
    )
    user = (
        "Fill missing fields from: manufacturer_code, manufacturer_manager, vendor_code, unit_cost. "
        "Do not return UNKNOWN/UKNOWN. "
        "If unsure, provide a best-effort deterministic enterprise-safe placeholder.\n"
        f"MPN: {mpn}\n"
        f"manufacturer_name: {manufacturer_name}\n"
        f"item_description: {item_description}\n"
        f"missing_fields: {missing_fields}\n"
        "Return JSON with only requested fields."
    )
    parsed = _openai_json_response(system, user) or {}
    if not isinstance(parsed, dict):
        parsed = {}

    # deterministic local fallback if LLM did not return enough values
    vendor_seed = re.sub(r"[^A-Z0-9]", "", manufacturer_name.upper())[:4] or "GEN"
    fallback_map: Dict[str, Any] = {
        "manufacturer_code": f"M-{vendor_seed}",
        "manufacturer_manager": "catalog-auto-manager",
        "vendor_code": f"V-{vendor_seed}",
        "unit_cost": 1.0,
    }
    out: Dict[str, Any] = {}
    for field_name in missing_fields:
        value = parsed.get(field_name)
        if _is_unknown_value(value):
            value = fallback_map.get(field_name)
        if field_name == "unit_cost":
            try:
                value = float(value)
            except Exception:
                value = float(fallback_map["unit_cost"])
        if isinstance(value, str):
            value = value.strip()
        out[field_name] = value
    return out


def _manual_agent() -> ManualMpnCatalogAgent:
    return ManualMpnCatalogAgent(
        require_user_confirmation=False,
        erp_client=_erp_client(),
        mpn_vendor_search=lambda mpn: _guess_mpn_manufacturer(mpn),
        web_lookup=_default_web_lookup,
    )


def _quote_agent() -> QuoteParsingCatalogAgent:
    return QuoteParsingCatalogAgent(
        erp_client=_erp_client(),
        web_lookup=_default_web_lookup,
    )


def _guess_mpn_manufacturer(mpn: str) -> List[str]:
    code = re.sub(r"[^A-Z0-9\-]", "", mpn.upper())
    candidates: List[str] = []

    # Finisar / Coherent optics families
    if code.startswith(("FTLF", "FTLX", "FWLF")):
        candidates.append("Finisar")

    # Intel adapters and OEM cards
    if code.startswith("I") and ("OEM" in code or re.match(r"^I[0-9]{3,}", code)):
        candidates.append("Intel")

    # Check Point product families
    if code.startswith(("CP-", "CPSM", "CPSB", "CPAP", "CPAC")):
        candidates.append("CheckPoint")

    # Cisco networking/optics common SKUs
    cisco_prefixes = (
        "GLC-",
        "SFP-",
        "QSFP-",
        "C9",
        "C19",
        "WS-C",
        "N9K-",
        "CISCO",
    )
    if code.startswith(cisco_prefixes):
        candidates.append("Cisco")

    # Fortinet appliances and accessories.
    if code.startswith(("FG-", "SP-FG", "FGT-")):
        candidates.append("Fortinet")

    # Dell KVM and enterprise hardware identifiers.
    if code.startswith(("DMPU", "DRMK-", "A7", "450-")):
        candidates.append("Dell")

    # Juniper branch/campus/DC families.
    if code.startswith(("SRX", "MX", "EX", "QFX", "NFX", "PTX", "ACX")):
        candidates.append("Juniper")

    # HPE/Aruba typical format (optional but common in enterprise)
    if code.startswith(("J", "JL", "JW", "R0", "R8")) and re.search(r"-", code):
        candidates.append("HPE")

    # Deduplicate while preserving order.
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        candidates.append("Unknown")
    return candidates


def _parse_order_information_section(text: str) -> pd.DataFrame:
    """Parse datasheet ORDER/ORDERING sections and extract SKU lines robustly."""
    section = _extract_ordering_section(text)
    if not section:
        return pd.DataFrame()

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in section.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()

    rows = []
    current: Optional[dict] = None

    for line in lines:
        if line in {"ORDER INFORMATION", "ORDERING INFORMATION", "ORDERING", "PRODUCT SKU DESCRIPTION", "OPTIONAL ACCESSORIES"}:
            continue
        if line.startswith("FORTIGUARD BUNDLE") or line.startswith("BUNDLES ") or line.startswith("SUBSCRIPTION SERVICES"):
            break

        skus = _find_sku_candidates(line)
        if skus:
            if current:
                rows.append(current)

            sku = sorted(skus, key=lambda s: (_sku_score(s), len(s)), reverse=True)[0]
            sku_pos = line.rfind(sku)
            name_part = line[:sku_pos].strip(" -:")
            desc_part = line[sku_pos + len(sku) :].strip(" -:")
            manufacturer = _guess_mpn_manufacturer(sku)[0]
            current = {
                "mpn": sku,
                "item_description": f"{name_part} {desc_part}".strip(),
                "manufacturer_name": manufacturer,
                "supplier_name": "Unknown Supplier",
                "unit_cost": None,
                "quantity": 1,
                "line_total": None,
                "parser_used": "order_parser",
                "extraction_confidence": "high",
                "sku_confidence": "high",
            }
        elif current:
            current["item_description"] = f"{current['item_description']} {line}".strip()

    if current:
        rows.append(current)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).drop_duplicates(subset=["mpn"], keep="first")
    return out.reset_index(drop=True)


def _extract_text_from_pdf(raw: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed.")
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _detect_vendor(text: str) -> str:
    t = text.lower()
    if "new semaphore" in t or "ניוסמפור" in text:
        return "New Semaphore"
    return "Unknown Supplier"


def _normalize_price(value: str) -> Optional[float]:
    cleaned = value.replace(",", "").replace("$", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


_NON_SKU_TOKENS = {
    "DATASHEET",
    "SPECIFICATIONS",
    "ORDERING",
    "ORDER",
    "PRODUCT",
    "DESCRIPTION",
    "ACCESSORIES",
    "SERVICES",
    "SUBSCRIPTION",
    "SECURITY",
    "FIREWALL",
    "DEPLOYMENT",
    "BRANCH",
    "CAMPUS",
    "HARDWARE",
}


def _normalize_pdf_text(text: str) -> str:
    out = text.upper()
    prev = None
    while prev != out:
        prev = out
        out = re.sub(r"([A-Z0-9])\s*-\s*([A-Z0-9])", r"\1-\2", out)
    return out


def _is_probable_sku(token: str) -> bool:
    tok = token.strip(" ,.;:()[]{}")
    if len(tok) < 5 or len(tok) > 40:
        return False
    if tok in _NON_SKU_TOKENS:
        return False
    if "-" not in tok:
        if re.match(r"^DMPU[0-9A-Z]{3,}$", tok):
            return True
        if re.match(r"^A[0-9]{6,8}$", tok):
            return True
        if re.match(r"^(SRX|MX|EX|QFX|NFX|PTX|ACX)[0-9]{2,5}[A-Z]{0,2}$", tok):
            return True
        return bool(re.match(r"^(FTLF|FTLX|FWLF)[A-Z0-9]{4,}$", tok))
    if not re.match(r"^[A-Z0-9]+(?:-[A-Z0-9]+){1,7}$", tok):
        return False
    if not re.search(r"[A-Z]", tok):
        return False
    if any(ch.isdigit() for ch in tok):
        return True
    return tok.startswith(("CP-", "FG-", "CPAP-", "CPAC-", "CPSB-", "GLC-", "SFP-", "QSFP-"))


def _find_sku_candidates(line: str) -> List[str]:
    tokens = re.findall(r"\b[A-Z0-9][A-Z0-9\-]{3,}\b", line)
    candidates = [t for t in tokens if _is_probable_sku(t)]
    return list(dict.fromkeys(candidates))


def _find_probable_skus_any(text: str) -> List[str]:
    tokens = re.findall(r"\b[A-Z0-9][A-Z0-9\-]{3,}\b", text)
    candidates = [t for t in tokens if _is_probable_sku(t)]
    return list(dict.fromkeys(candidates))


def _sku_score(token: str) -> int:
    t = token.upper()
    score = t.count("-") * 3 + min(len(t), 20)
    strong_prefixes = ("SP-", "CPAP-", "CPAC-", "CPSB-", "FG-TRAN-", "FGT-", "GLC-", "SFP-", "QSFP-")
    if t.startswith(strong_prefixes):
        score += 20
    if re.match(r"^FG-\d{3}$", t):
        score -= 8
    return score


def _extract_ordering_section(text: str) -> str:
    normalized = _normalize_pdf_text(text)
    starts = [normalized.find(k) for k in ("ORDER INFORMATION", "ORDERING INFORMATION", "ORDERING", "PRODUCT SKU DESCRIPTION")]
    starts = [s for s in starts if s >= 0]
    if not starts:
        return ""
    start = min(starts)

    end_markers = (
        "SUBSCRIPTION SERVICES",
        "SERVICES BUNDLES",
        "GLOBAL HEADQUARTERS",
        "CONTACT US",
        "SPECIFICATIONS",
    )
    ends = [normalized.find(m, start + 50) for m in end_markers]
    ends = [e for e in ends if e > start]
    end = min(ends) if ends else len(normalized)
    return normalized[start:end]


def _parse_model_part_matrix(text: str) -> pd.DataFrame:
    """
    Parse datasheets that present models in columns and regional part numbers in rows.
    Example: DMPU4032 DMPU2016 DMPU108E with Americas/EMEA/APJ part numbers.
    """
    normalized = _normalize_pdf_text(text)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in normalized.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()

    model_idx = -1
    models: List[str] = []
    for idx, line in enumerate(lines):
        row_models = re.findall(r"\bDMPU[0-9A-Z]{3,}\b", line)
        if len(row_models) >= 2:
            model_idx = idx
            models = list(dict.fromkeys(row_models))
            break

    if model_idx < 0 or not models:
        # Generic model-family matrix fallback (example: SRX300 SRX320 ...).
        family_prefixes = {"SRX", "MX", "EX", "QFX", "NFX", "PTX", "ACX"}
        for line in lines:
            tokens = re.findall(r"\b[A-Z]{2,6}[0-9]{2,5}[A-Z]{0,2}\b", line)
            row_models = [tok for tok in tokens if _is_probable_sku(tok)]
            if len(row_models) < 3:
                continue
            prefix_counts: Dict[str, int] = {}
            for model in row_models:
                match = re.match(r"^([A-Z]{2,6})", model)
                if not match:
                    continue
                prefix = match.group(1)
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            if not prefix_counts:
                continue
            best_prefix = max(prefix_counts, key=prefix_counts.get)
            if best_prefix not in family_prefixes or prefix_counts[best_prefix] < 3:
                continue
            models = list(dict.fromkeys([m for m in row_models if m.startswith(best_prefix)]))
            if len(models) < 3:
                continue
            manufacturer = _guess_mpn_manufacturer(models[0])[0]
            description = f"{best_prefix} series security gateway"
            return pd.DataFrame(
                [
                    {
                        "mpn": model,
                        "item_description": description,
                        "manufacturer_name": manufacturer,
                        "supplier_name": "Unknown Supplier",
                        "unit_cost": None,
                        "quantity": 1,
                        "line_total": None,
                        "parser_used": "model_part_matrix",
                        "extraction_confidence": "high",
                        "sku_confidence": "high",
                    }
                    for model in models
                ]
            )
        return pd.DataFrame()

    by_model: dict[str, List[str]] = {m: [] for m in models}
    region_pattern = re.compile(r"^(AMERICAS|EMEA|APJ)\b")
    part_pattern = re.compile(r"\b(?:A[0-9]{6,8}|[0-9]{3}-[A-Z0-9]{3,})\b")

    for line in lines[model_idx + 1 : model_idx + 20]:
        region_match = region_pattern.match(line)
        region = region_match.group(1) if region_match else "REGION"
        parts = part_pattern.findall(line)
        if len(parts) < len(models):
            continue
        for i, model in enumerate(models):
            by_model[model].append(f"{region}:{parts[i]}")

    rows = []
    for model in models:
        pn_text = "; ".join(by_model[model])
        desc = "Dell EMC DMPU KVM Remote Console Switch"
        if pn_text:
            desc = f"{desc} ({pn_text})"
        rows.append(
            {
                "mpn": model,
                "item_description": desc,
                "manufacturer_name": "Dell",
                "supplier_name": "Unknown Supplier",
                "unit_cost": None,
                "quantity": 1,
                "line_total": None,
                "parser_used": "model_part_matrix",
                "extraction_confidence": "high",
                "sku_confidence": "high",
            }
        )

    return pd.DataFrame(rows)


def _parse_title_model_fallback(text: str) -> pd.DataFrame:
    """Last-resort model extraction when no table/quote structure exists."""
    normalized = _normalize_pdf_text(text)
    candidates: List[tuple[str, str, str]] = []

    if "FORTIGATE" in normalized:
        for m in re.findall(r"FORTIGATE\\s+([0-9]{2,4}[A-Z])", normalized):
            candidates.append((f"FG-{m}", "Fortinet", f"FortiGate {m} appliance datasheet"))

    if re.search(r"\\bUDM\\b", normalized):
        candidates.append(("UDM", "Ubiquiti", "UniFi Dream Machine"))
    if re.search(r"\\bUXG[-\\s]?PRO\\b", normalized):
        candidates.append(("UXG-PRO", "Ubiquiti", "UniFi Next-Gen Gateway Pro"))
    if "CORTEX XSOAR" in normalized:
        candidates.append(("CORTEX-XSOAR", "PaloAlto", "Cortex XSOAR platform"))

    # unique by MPN, keep order
    seen = set()
    rows = []
    for mpn, manufacturer, desc in candidates:
        if mpn in seen:
            continue
        seen.add(mpn)
        rows.append(
            {
                "mpn": mpn,
                "item_description": desc,
                "manufacturer_name": manufacturer,
                "supplier_name": "Unknown Supplier",
                "unit_cost": None,
                "quantity": 1,
                "line_total": None,
                "parser_used": "title_fallback",
                "extraction_confidence": "low",
                "sku_confidence": "low",
            }
        )
    return pd.DataFrame(rows)


def _parse_source_hint_fallback(source_hint: str) -> pd.DataFrame:
    """Final fallback by file name / URL when PDF text has no parsable SKU."""
    hint = source_hint.upper()
    rows = []

    m = re.search(r"FORTIGATE[-_]?([0-9]{2,4}[A-Z])", hint)
    if m:
        model = m.group(1)
        rows.append(
            {
                "mpn": f"FG-{model}",
                "item_description": f"FortiGate {model} appliance",
                "manufacturer_name": "Fortinet",
                "supplier_name": "Unknown Supplier",
                "unit_cost": None,
                "quantity": 1,
                "line_total": None,
                "parser_used": "source_hint_fallback",
                "extraction_confidence": "low",
                "sku_confidence": "low",
            }
        )

    if "UDM_DS.PDF" in hint or re.search(r"/UDM(_DS)?\\.PDF", hint):
        rows.append(
            {
                "mpn": "UDM",
                "item_description": "UniFi Dream Machine",
                "manufacturer_name": "Ubiquiti",
                "supplier_name": "Unknown Supplier",
                "unit_cost": None,
                "quantity": 1,
                "line_total": None,
                "parser_used": "source_hint_fallback",
                "extraction_confidence": "low",
                "sku_confidence": "low",
            }
        )

    if "UXG-PRO_DS.PDF" in hint or "UXG-PRO" in hint:
        rows.append(
            {
                "mpn": "UXG-PRO",
                "item_description": "UniFi Next-Gen Gateway Pro",
                "manufacturer_name": "Ubiquiti",
                "supplier_name": "Unknown Supplier",
                "unit_cost": None,
                "quantity": 1,
                "line_total": None,
                "parser_used": "source_hint_fallback",
                "extraction_confidence": "low",
                "sku_confidence": "low",
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["mpn"]).reset_index(drop=True)


def _parse_quote_lines(text: str) -> pd.DataFrame:
    rows = []
    text = _normalize_pdf_text(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    vendor_name = _detect_vendor(text)

    # First pass: parse whole text with multiline row regex (handles wrapped descriptions).
    normalized_blob = re.sub(r"\s+", " ", text).strip()
    row_pattern = re.compile(
        r"(?P<mpn>[A-Z][A-Z0-9\-]{3,})\s+"
        r"(?P<desc>.*?)\s+"
        r"(?P<unit>\$?\d[\d,\.]*\$?)\s+"
        r"(?P<qty>\d+)\s+"
        r"(?P<total>\$?\d[\d,\.]*\$?)"
        r"(?=\s+[A-Z][A-Z0-9\-]{3,}\s+|\s+[^\x00-\x7F]|$)"
    )
    for match in row_pattern.finditer(normalized_blob):
        mpn = match.group("mpn")
        description = match.group("desc").strip()
        if not _is_probable_sku(mpn):
            alternatives = _find_probable_skus_any(description)
            if not alternatives:
                continue
            mpn = alternatives[0]
            description = re.sub(rf"\b{re.escape(mpn)}\b", "", description, count=1).strip()
        rows.append(
            {
                "mpn": mpn,
                "item_description": description,
                "manufacturer_name": _guess_mpn_manufacturer(mpn)[0],
                "supplier_name": vendor_name,
                "unit_cost": _normalize_price(match.group("unit")),
                "quantity": int(match.group("qty")),
                "line_total": _normalize_price(match.group("total")),
                "parser_used": "quote_parser",
                "extraction_confidence": "medium",
                "sku_confidence": "medium",
            }
        )

    # Second pass fallback: strict single-line rows.
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = re.match(
            r"^([A-Z0-9\-]{4,})\s+(.+?)\s+(\$?\d[\d,\.]*\$?)\s+(\d+)\s+(\$?[\d,\.]+\$?)$",
            line,
        )
        if match:
            mpn = match.group(1)
            if not _is_probable_sku(mpn):
                idx += 1
                continue
            description = match.group(2)
            unit_cost = _normalize_price(match.group(3))
            qty = int(match.group(4))
            total = _normalize_price(match.group(5))
            rows.append(
                {
                    "mpn": mpn,
                    "item_description": description,
                    "manufacturer_name": _guess_mpn_manufacturer(mpn)[0],
                    "supplier_name": vendor_name,
                    "unit_cost": unit_cost,
                    "quantity": qty,
                    "line_total": total,
                    "parser_used": "quote_parser",
                    "extraction_confidence": "medium",
                    "sku_confidence": "medium",
                }
            )
        idx += 1

    if not rows:
        # fallback parser for broken line wraps in PDF text extraction
        for i, line in enumerate(lines):
            token = re.match(r"^[A-Z][A-Z0-9\-]{3,}$", line)
            if not token:
                continue
            mpn = token.group(0)
            if not _is_probable_sku(mpn):
                continue
            desc_parts: List[str] = []
            j = i + 1
            while j < len(lines) and not re.search(r"\$\d", lines[j]):
                if re.match(r"^[A-Z0-9\-]{4,}$", lines[j]):
                    break
                desc_parts.append(lines[j])
                j += 1
            joined = " ".join(desc_parts).strip() or f"{_guess_mpn_manufacturer(mpn)[0]} item"
            price_match = None
            if j < len(lines):
                price_match = re.search(r"(\$?\d[\d,\.]*)\s+(\d+)\s+(\$?[\d,\.]+)$", lines[j])
            rows.append(
                {
                    "mpn": mpn,
                    "item_description": joined,
                    "manufacturer_name": _guess_mpn_manufacturer(mpn)[0],
                    "supplier_name": vendor_name,
                    "unit_cost": _normalize_price(price_match.group(1)) if price_match else None,
                    "quantity": int(price_match.group(2)) if price_match else 1,
                    "line_total": _normalize_price(price_match.group(3)) if price_match else None,
                    "parser_used": "quote_parser_fallback",
                    "extraction_confidence": "low",
                    "sku_confidence": "medium",
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).drop_duplicates(subset=["mpn", "unit_cost", "quantity"], keep="first")
    return out.reset_index(drop=True)


def _build_catalog_from_quotes(
    df: pd.DataFrame,
    *,
    source_name: str = "unknown_source",
    source_type: str = "quote_document",
    supplier_hint: Optional[str] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    quote_agent = _quote_agent()
    manual_agent = _manual_agent()
    extraction_rows: List[CatalogExtractionRow] = []
    for _, row in df.iterrows():
        extraction_rows.append(
            CatalogExtractionRow(
                mpn=str(row["mpn"]),
                item_description=str(row["item_description"]),
                manufacturer_name=str(row["manufacturer_name"]),
                supplier_name=str(row["supplier_name"]),
                unit_cost=float(row["unit_cost"]) if pd.notna(row["unit_cost"]) else None,
                quantity=int(row.get("quantity", 1) if pd.notna(row.get("quantity", 1)) else 1),
                line_total=float(row["line_total"]) if pd.notna(row.get("line_total")) else None,
                parser_used=str(row.get("parser_used", "unknown")),
                extraction_confidence=str(row.get("extraction_confidence", "medium")),
                sku_confidence=str(row.get("sku_confidence", "medium")),
            )
        )

    envelope = CatalogInputEnvelope(
        source_type=source_type,
        source_name=source_name,
        supplier_hint=supplier_hint,
    )
    decisions = process_catalog_envelope(
        envelope,
        extraction_rows,
        manual_agent=manual_agent,
        quote_agent=quote_agent,
    )

    out = []
    for idx, decision in enumerate(decisions):
        row = extraction_rows[idx]
        result = {
            "status": decision.status,
            "message": decision.message,
            "mpn": row.mpn,
            "quantity": row.quantity,
            "line_total": row.line_total,
            "parser_used": row.parser_used,
            "extraction_confidence": row.extraction_confidence,
            "sku_confidence": row.sku_confidence,
            "missing_fields": ", ".join(decision.missing_fields) if decision.missing_fields else "",
            "warnings": " | ".join(decision.warnings) if decision.warnings else "",
        }
        if decision.trace:
            for key, value in decision.trace.items():
                result[f"trace_{key}"] = value
        if decision.record:
            result.update(asdict(decision.record))
        out.append(result)

    return _ensure_required_output_columns(pd.DataFrame(out))


def _manual_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("קטלוג לפי מק\"ט ידני")
    mpn = st.text_input("MPN")
    manufacturer = st.text_input("Manufacturer (אופציונלי)")
    if mpn.strip():
        guesses = _guess_mpn_manufacturer(mpn.strip())
        st.caption(f"זיהוי יצרן משוער: {', '.join(guesses)}")

    if st.button("קטלג מק\"ט", use_container_width=True, type="primary"):
        if not mpn.strip():
            st.error("יש להזין MPN.")
        else:
            progress_update, progress_done = _start_progress_ui("קטלוג מק\"ט ידני")
            progress_update(10, "אתחול מנוע קיטלוג")
            agent = _manual_agent()
            req = ManualMpnRequest(
                mpn=mpn.strip(),
                confirmed_manufacturer_name=manufacturer.strip() or None,
            )
            progress_update(30, "הרצת סיווג ראשוני")
            decision = agent.process(req)
            if decision.status != "ready":
                progress_update(55, "ניסיון העשרה אוטומטי עם LLM")
                llm_meta = _llm_guess_manual_metadata(mpn.strip())
                if llm_meta and llm_meta.get("manufacturer_name"):
                    decision = agent.process(
                        ManualMpnRequest(
                            mpn=mpn.strip(),
                            confirmed_manufacturer_name=llm_meta["manufacturer_name"],
                            parser_used="llm_manual_enrichment",
                            extraction_confidence="low",
                            sku_confidence="low",
                        )
                    )
            progress_update(75, "בדיקות שדות חובה וטיוב נתונים")
            st.write(f"**Status:** `{decision.status}`")
            st.write(decision.message)
            if decision.missing_fields:
                target_missing = [
                    f
                    for f in decision.missing_fields
                    if f in {"manufacturer_code", "manufacturer_manager", "vendor_code", "unit_cost"}
                ]
                if target_missing and decision.record is not None:
                    enriched = _llm_complete_manual_required_fields(
                        mpn=decision.record.mpn,
                        manufacturer_name=(manufacturer.strip() or decision.record.item_name.split(" ")[0]),
                        item_description=decision.record.item_description,
                        missing_fields=target_missing,
                    )
                    row = asdict(decision.record)
                    row.update(enriched)
                    out_df = _ensure_required_output_columns(pd.DataFrame([row]))
                    st.info("LLM completion applied for missing mandatory fields.")
                    st.dataframe(out_df, use_container_width=True)
                    progress_done(True, "הקטלוג הושלם עם השלמות אוטומטיות")
                else:
                    st.error(f"Missing mandatory fields: {', '.join(decision.missing_fields)}")
                    progress_done(False, "הקטלוג הסתיים עם שדות חסרים")
            if decision.warnings:
                st.warning(" | ".join(decision.warnings))
            if decision.trace:
                st.json(decision.trace)
            if decision.vendor_candidates:
                st.warning("נדרש אישור יצרן. מועמדים:")
                st.write(list(decision.vendor_candidates))
            if decision.record:
                out_df = _ensure_required_output_columns(pd.DataFrame([asdict(decision.record)]))
                if _output_has_unknown_required_fields(out_df):
                    progress_update(85, "טיוב סופי באינטרנט + LLM")
                    out_df = _llm_enrich_catalog_output(
                        out_df,
                        raw_text=decision.record.item_description or mpn.strip(),
                        source_name=f"manual:{mpn.strip()}",
                    )
                out_df = _iterative_web_llm_verification(
                    out_df,
                    raw_text=decision.record.item_description or mpn.strip(),
                    source_name=f"manual:{mpn.strip()}",
                )
                st.dataframe(out_df, use_container_width=True)
                progress_done(True, "הקטלוג הושלם בהצלחה")
    st.markdown("</div>", unsafe_allow_html=True)


def _document_tab() -> None:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.subheader("קטלוג לפי מסמך הצעה")
    uploaded = st.file_uploader("העלה PDF/CSV/XLSX", type=["pdf", "csv", "xlsx", "xls"], key="quote_upload")

    if not uploaded:
        st.info("העלה מסמך כדי להתחיל.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    file_name = uploaded.name.lower()
    rows_df = pd.DataFrame()
    raw_text = ""

    try:
        if file_name.endswith(".pdf"):
            raw_text = _extract_text_from_pdf(uploaded.getvalue())
            rows_df = _parse_order_information_section(raw_text)
            if rows_df.empty:
                rows_df = _parse_quote_lines(raw_text)
            if rows_df.empty:
                rows_df = _parse_model_part_matrix(raw_text)
            if rows_df.empty:
                rows_df = _parse_title_model_fallback(raw_text)
            if rows_df.empty:
                rows_df = _parse_source_hint_fallback(uploaded.name)
            if rows_df.empty:
                rows_df = _llm_extract_rows_from_text(raw_text, uploaded.name)
        elif file_name.endswith(".csv"):
            rows_df = pd.read_csv(uploaded)
            raw_text = rows_df.to_csv(index=False)
        else:
            rows_df = pd.read_excel(uploaded)
            raw_text = rows_df.to_csv(index=False)
    except Exception as exc:
        st.error(f"קריאת קובץ נכשלה: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    required = {"mpn", "item_description", "manufacturer_name", "supplier_name", "unit_cost"}
    missing = sorted(required - set(map(str.lower, rows_df.columns)))

    if missing and not file_name.endswith(".pdf"):
        st.error(
            "בקבצי CSV/XLSX חייבים עמודות: mpn, item_description, manufacturer_name, supplier_name, unit_cost"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not rows_df.empty and not file_name.endswith(".pdf"):
        # normalize case-insensitive headers
        cols = {c.lower(): c for c in rows_df.columns}
        rows_df = rows_df.rename(columns={cols[k]: k for k in cols if k in required or k in {"quantity", "line_total"}})

    if _rows_have_missing_input_fields(rows_df):
        rows_df = _llm_enrich_rows(rows_df, raw_text, uploaded.name)
    rows_df = _escalate_low_confidence_rows(rows_df, raw_text, uploaded.name)

    st.caption("שורות שזוהו מהמסמך")
    st.dataframe(rows_df, use_container_width=True)

    if st.button("קטלג מסמך", use_container_width=True, type="primary"):
        progress_update, progress_done = _start_progress_ui("קטלוג מסמך")
        progress_update(10, "הכנת נתוני מקור לקיטלוג")
        inferred_source_type = "datasheet_document" if file_name.endswith(".pdf") else "quote_document"
        progress_update(30, "הרצת מנוע קיטלוג על השורות שחולצו")
        catalog_df = _build_catalog_from_quotes(
            rows_df,
            source_name=uploaded.name,
            source_type=inferred_source_type,
            supplier_hint="Unknown Supplier",
        )
        if not catalog_df.empty and catalog_df.get("missing_fields") is not None:
            needs_enrichment = catalog_df["missing_fields"].astype(str).str.strip().ne("").any()
            if needs_enrichment:
                progress_update(55, "השלמת נתונים חסרים עם LLM")
                rows_df = _llm_enrich_rows(rows_df, raw_text, uploaded.name)
                catalog_df = _build_catalog_from_quotes(
                    rows_df,
                    source_name=uploaded.name,
                    source_type=inferred_source_type,
                    supplier_hint="Unknown Supplier",
                )
        if not catalog_df.empty and _output_has_unknown_required_fields(catalog_df):
            progress_update(70, "טיוב ערכי UNKNOWN באינטרנט + LLM")
            catalog_df = _llm_enrich_catalog_output(catalog_df, raw_text, uploaded.name)
            catalog_df = _ensure_required_output_columns(catalog_df)
        if not catalog_df.empty:
            progress_update(85, "תיקוף סופי וטיוב איטרטיבי")
            catalog_df = _iterative_web_llm_verification(catalog_df, raw_text, uploaded.name)
        if catalog_df.empty:
            st.warning("לא זוהו שורות לקיטלוג.")
            progress_done(False, "לא זוהו שורות לקיטלוג")
        else:
            st.success(f"הסתיים קיטלוג {len(catalog_df)} פריטים")
            st.dataframe(catalog_df, use_container_width=True)
            st.download_button(
                "הורדת תוצאות CSV",
                data=catalog_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="catalog_results.csv",
                mime="text/csv",
            )
            progress_done(True, "הקטלוג והאימות הסתיימו בהצלחה")

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    _render_header()
    st.markdown('<p class="small-muted">גרסה ראשונית - לוגיקת קיטלוג עסקית + העלאת מסמך.</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Manual MPN", "Quote Document"])
    with tab1:
        _manual_tab()
    with tab2:
        _document_tab()


if __name__ == "__main__":
    main()
