# Catalog Agent UI

## Run

```powershell
cd workspaces\ws_20260216-190210\project
c:\dev\my-python\gpt-agent\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
c:\dev\my-python\gpt-agent\.venv\Scripts\python.exe -m streamlit run catalog_ui.py
```

## What it does

- Manual MPN flow (with manufacturer confirmation behavior)
- Quote document flow (PDF/CSV/XLSX)
- Auto extraction of quote lines from PDF text
- Catalog decision output table (`ready`, `needs_user_confirmation`, `needs_manual_review`)
- CSV export of catalog results
- Strict mandatory-field validation (`ready` only if all mandatory fields are present)
- Trace output per row (`parser_used`, confidence, ERP lookup outcomes)

## ERP Live API (optional)

When set, the app uses ERP HTTP lookups for manufacturer/vendor enrichment:

```powershell
$env:ERP_API_BASE_URL = "https://erp.example.com/catalog"
$env:ERP_API_TOKEN = "your-token"
```

Expected endpoints:

- `GET /manufacturer/code?vendor_name=...`
- `GET /manufacturer/manager?vendor_name=...`
- `GET /vendor/code?supplier_name=...`
- `GET /vendor/preferred?mpn=...&manufacturer_name=...&since_days=365`

## LLM Fallback (optional)

If extraction fails or mandatory inputs are missing, the app can call an LLM for:

- MPN metadata guess (manual flow)
- Row extraction from document text
- Missing-field enrichment for extracted rows
- Output-level enrichment when required output fields contain `UNKNOWN` / `UKNOWN`
- Manual-MPN specific completion for missing mandatory fields:
  - `manufacturer_code`, `manufacturer_manager`, `vendor_code`, `unit_cost`
- Final `No-None Guarantee` pass:
  - if any required output field is `None`/`NaN`/`NONE`/`UNKNOWN`, app performs web-context search + LLM completion
  - if still missing, deterministic fallback values are applied so no required field remains empty
- Low-confidence escalation:
  - every row with `extraction_confidence=low` is re-checked using web context + LLM up to multiple rounds
  - preference is given to official vendor domains (`site:vendor.com`)
- End-of-catalog validation loop:
  - after cataloging, app performs iterative cross-validation with web + LLM until `verification_confidence=HIGH`

Configure:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_MODEL = "gpt-4.1-mini"
```

## Output Columns (Spec Required)

The app enforces these columns in every catalog output:

- `mpn`, `item_name`, `item_description`
- `level_1`, `level_2`, `level_3`, `level_4`
- `software_item`, `inventory_managed`, `serialized`, `subscription`
- `manufacturer_code`, `manufacturer_manager`
- `vendor_code`, `vendor_part_number`, `unit_cost`
- `weight_kg`, `height_cm`, `width_cm`, `depth_cm`, `datasheet_link`

## CSV/XLSX expected columns

- `mpn`
- `item_description`
- `manufacturer_name`
- `supplier_name`
- `unit_cost`
- optional: `quantity`, `line_total`
