"""App package exports."""

__version__ = "0.3.0"

from app.catalog_agent import (
    CatalogExtractionRow,
    CatalogInputEnvelope,
    CatalogDecision,
    CatalogRecord,
    ERPHttpClient,
    ERPClient,
    InMemoryERPClient,
    ManualMpnCatalogAgent,
    ManualMpnRequest,
    PurchaseHistoryRecord,
    QuoteParsingCatalogAgent,
    QuoteRequest,
    ValidationResult,
    WebCatalogHint,
    process_catalog_envelope,
    validate_mandatory_fields,
)
from app.parser import DataParser, Parser, TextParser

__all__ = [
    "CatalogDecision",
    "CatalogRecord",
    "CatalogExtractionRow",
    "CatalogInputEnvelope",
    "ERPClient",
    "ERPHttpClient",
    "InMemoryERPClient",
    "ManualMpnCatalogAgent",
    "ManualMpnRequest",
    "PurchaseHistoryRecord",
    "QuoteParsingCatalogAgent",
    "QuoteRequest",
    "ValidationResult",
    "WebCatalogHint",
    "process_catalog_envelope",
    "validate_mandatory_fields",
    "Parser",
    "TextParser",
    "DataParser",
]
