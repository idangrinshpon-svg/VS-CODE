"""Business-logic catalog agents for manual MPN entry and quote parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import re
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

CategoryPath = Tuple[str, Optional[str], Optional[str], Optional[str]]


@dataclass(frozen=True)
class ManualMpnRequest:
    mpn: str
    manufacturer_name: Optional[str] = None
    confirmed_manufacturer_name: Optional[str] = None
    parser_used: str = "manual_mpn"
    extraction_confidence: str = "high"
    sku_confidence: str = "high"


@dataclass(frozen=True)
class QuoteRequest:
    mpn: str
    item_description: str
    manufacturer_name: str
    supplier_name: str
    unit_cost: Optional[float]
    vendor_part_number: Optional[str] = None
    parser_used: str = "quote_parser"
    extraction_confidence: str = "medium"
    sku_confidence: str = "medium"


@dataclass(frozen=True)
class CatalogInputEnvelope:
    source_type: str  # manual_mpn | quote_document | datasheet_document
    source_name: str
    raw_text: Optional[str] = None
    file_reference: Optional[str] = None
    supplier_hint: Optional[str] = None


@dataclass(frozen=True)
class CatalogExtractionRow:
    mpn: str
    item_description: str
    manufacturer_name: str
    supplier_name: str
    unit_cost: Optional[float]
    quantity: int = 1
    line_total: Optional[float] = None
    parser_used: str = "unknown"
    extraction_confidence: str = "medium"
    sku_confidence: str = "medium"


@dataclass(frozen=True)
class PurchaseHistoryRecord:
    purchased_on: date
    vendor_code: str
    manufacturer_name: Optional[str] = None
    mpn: Optional[str] = None


@dataclass(frozen=True)
class WebCatalogHint:
    manufacturer_name: Optional[str] = None
    item_description: Optional[str] = None
    category_hint: Optional[CategoryPath] = None
    item_kind: Optional[str] = None  # hardware | software | service
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    width_cm: Optional[float] = None
    depth_cm: Optional[float] = None
    datasheet_link: Optional[str] = None


@dataclass(frozen=True)
class CatalogRecord:
    mpn: str
    item_name: str
    item_description: str
    level_1: str
    level_2: Optional[str]
    level_3: Optional[str]
    level_4: Optional[str]
    software_item: bool
    inventory_managed: bool
    serialized: bool
    subscription: bool
    manufacturer_code: Optional[str]
    manufacturer_manager: Optional[str]
    vendor_code: Optional[str]
    vendor_part_number: str
    unit_cost: Optional[float]
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    width_cm: Optional[float] = None
    depth_cm: Optional[float] = None
    datasheet_link: Optional[str] = None
    category_resolution_reason: str = "fallback:oem"


@dataclass(frozen=True)
class CatalogDecision:
    status: str  # ready | needs_user_confirmation | needs_manual_review
    message: str
    record: Optional[CatalogRecord] = None
    vendor_candidates: Sequence[str] = field(default_factory=tuple)
    missing_fields: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
    trace: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    missing_fields: Sequence[str] = field(default_factory=tuple)


class ERPClient(Protocol):
    def get_manufacturer_code(self, vendor_name: str) -> Optional[str]:
        ...

    def get_manufacturer_manager(self, vendor_name: str) -> Optional[str]:
        ...

    def get_vendor_code(self, supplier_name: str) -> Optional[str]:
        ...

    def get_preferred_vendor(self, mpn: str, manufacturer_name: str, since_days: int = 365) -> Optional[str]:
        ...


class InMemoryERPClient:
    def __init__(
        self,
        manufacturer_code_lookup: Optional[Dict[str, str]] = None,
        manufacturer_manager_lookup: Optional[Dict[str, str]] = None,
        vendor_code_lookup: Optional[Dict[str, str]] = None,
        purchase_history: Sequence[PurchaseHistoryRecord] = (),
    ) -> None:
        self._manufacturer_code_lookup = {_norm(k): v for k, v in (manufacturer_code_lookup or {}).items()}
        self._manufacturer_manager_lookup = {_norm(k): v for k, v in (manufacturer_manager_lookup or {}).items()}
        self._vendor_code_lookup = {_norm(k): v for k, v in (vendor_code_lookup or {}).items()}
        self._purchase_history = list(purchase_history)

    def get_manufacturer_code(self, vendor_name: str) -> Optional[str]:
        return self._manufacturer_code_lookup.get(_norm(vendor_name))

    def get_manufacturer_manager(self, vendor_name: str) -> Optional[str]:
        return self._manufacturer_manager_lookup.get(_norm(vendor_name))

    def get_vendor_code(self, supplier_name: str) -> Optional[str]:
        return self._vendor_code_lookup.get(_norm(supplier_name))

    def get_preferred_vendor(self, mpn: str, manufacturer_name: str, since_days: int = 365) -> Optional[str]:
        ref = date.today()
        cutoff = ref - timedelta(days=since_days)
        matches = [
            h
            for h in self._purchase_history
            if h.purchased_on >= cutoff
            and (
                (h.mpn and _norm(h.mpn) == _norm(mpn))
                or (
                    h.manufacturer_name
                    and _norm(h.manufacturer_name) == _norm(manufacturer_name)
                )
            )
        ]
        if not matches:
            return None
        counts: Dict[str, int] = {}
        for record in matches:
            counts[record.vendor_code] = counts.get(record.vendor_code, 0) + 1
        return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


class ERPHttpClient:
    """
    Minimal ERP adapter over HTTP.
    Expected endpoints:
      GET {base_url}/manufacturer/code?vendor_name=...
      GET {base_url}/manufacturer/manager?vendor_name=...
      GET {base_url}/vendor/code?supplier_name=...
      GET {base_url}/vendor/preferred?mpn=...&manufacturer_name=...&since_days=...
    Response JSON: {"value": "..."} or {"vendor_code": "..."}.
    """

    def __init__(self, base_url: str, token: Optional[str] = None, timeout_seconds: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def _get_value(self, path: str, params: Dict[str, str]) -> Optional[str]:
        try:
            import requests  # local import to avoid hard dependency in all runtime modes
        except Exception:
            return None

        headers: Dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            response = requests.get(
                f"{self.base_url}{path}",
                params=params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
        except Exception:
            return None

        for key in ("value", "vendor_code", "manufacturer_code", "manager"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def get_manufacturer_code(self, vendor_name: str) -> Optional[str]:
        return self._get_value("/manufacturer/code", {"vendor_name": vendor_name})

    def get_manufacturer_manager(self, vendor_name: str) -> Optional[str]:
        return self._get_value("/manufacturer/manager", {"vendor_name": vendor_name})

    def get_vendor_code(self, supplier_name: str) -> Optional[str]:
        return self._get_value("/vendor/code", {"supplier_name": supplier_name})

    def get_preferred_vendor(self, mpn: str, manufacturer_name: str, since_days: int = 365) -> Optional[str]:
        return self._get_value(
            "/vendor/preferred",
            {
                "mpn": mpn,
                "manufacturer_name": manufacturer_name,
                "since_days": str(since_days),
            },
        )


VALID_CATEGORY_PATHS: Sequence[CategoryPath] = (
    ("Automic", None, None, None),
    ("Bynet Cloud IL", "Cloud Storage", None, None),
    ("Bynet Cloud IL", "Compute", None, None),
    ("Bynet Cloud IL", "MS365", None, None),
    ("Bynet Cloud IL", "Network", None, None),
    ("Bynet Cloud IL", "Security", None, None),
    ("Bynet Cloud IL", "Storage", "Tier - STD", None),
    ("Bynet Cloud IL", "Storage", None, None),
    ("Bynet Cloud IL", "Virtual Machines", None, None),
    ("Bynet Cloud IL", None, None, None),
    ("CAAS", None, None, None),
    ("Collaboration", "AV", "AUDIO VIDEO", None),
    ("Collaboration", "AV", "CONTROL", None),
    ("Collaboration", "AV", None, None),
    ("Collaboration", "CC", None, None),
    ("Collaboration", "PS", None, None),
    ("Collaboration", "SP", "BORDER CTRL", None),
    ("Collaboration", "SP", "RCS SW MSG SIG", None),
    ("Collaboration", "UCC", "BORDER CTRL", None),
    ("Collaboration", "UCC", "CC", None),
    ("Collaboration", "UCC", "INFRA", None),
    ("Collaboration", "UCC", "UC", None),
    ("Collaboration", "UCC", "UCC APPS", None),
    ("Collaboration", "UCC", "VC", None),
    ("Collaboration", "UCC", None, None),
    ("Collaboration", "VC", None, None),
    ("Collaboration", None, None, None),
    ("COMPUTERS", "Laptops", None, None),
    ("COMPUTERS", "PC", None, None),
    ("COMPUTERS", "Peripherals", None, None),
    ("COMPUTERS", "Servers", None, None),
    ("COMPUTERS", "Storage", None, None),
    ("COMPUTERS", None, None, None),
    ("Computing", "GPU", None, None),
    ("Computing", "Software", "Backup", None),
    ("Computing", "Software", "OS", None),
    ("Computing", "Software", "Virtualization", None),
    ("Computing", None, None, None),
    ("Data Center", "Cooling", None, None),
    ("Data Center", "Power", None, None),
    ("Data Center", "Racks", None, None),
    ("Hosting", "Domain", None, None),
    ("Hosting", "SSL", None, None),
    ("Hosting", "Web", None, None),
    ("Mobility & IoT", "IoT", None, None),
    ("Mobility & IoT", "Mobile Devices", None, None),
    ("Mobility & IoT", None, None, None),
    ("Networking", "Data Center", None, None),
    ("Networking", "Enterprise Network", "Access", None),
    ("Networking", "Enterprise Network", "Core", None),
    ("Networking", "Enterprise Network", "Wireless", None),
    ("Networking", "Industrial", None, None),
    ("Networking", "Optical", None, None),
    ("Networking", "Service Provider", None, None),
    ("Networking", None, None, None),
    ("OEM", None, None, None),
    ("SECURITY", "DDOS", "Cloud", None),
    ("SECURITY", "DDOS", "On-Prem", None),
    ("SECURITY", "DDOS", None, None),
    ("SECURITY", "Deceptor", None, None),
    ("SECURITY", "DLP", "Endpoint", None),
    ("SECURITY", "DLP", None, None),
    ("SECURITY", "DNS Protection", None, None),
    ("SECURITY", "ENCRYPTION", None, None),
    ("SECURITY", "EPM", None, None),
    ("SECURITY", "EPS", "Traditional", None),
    ("SECURITY", "EPS", None, None),
    ("SECURITY", "FIRE WALL/VPN", "CHEKPOINT", None),
    ("SECURITY", "FIRE WALL/VPN", None, None),
    ("SECURITY", "Firewall", "Branch", None),
    ("SECURITY", "Firewall", "Datacenter", None),
    ("SECURITY", "Firewall", "License, Support, Accessories", None),
    ("SECURITY", "Firewall", "Management", None),
    ("SECURITY", "Firewall", "Perimeter", None),
    ("SECURITY", "Firewall", None, None),
    ("SECURITY", "IAM/IDP", None, None),
    ("SECURITY", "Intelegance", None, None),
    ("SECURITY", "IPS", None, None),
    ("SECURITY", "KABLAN", None, None),
    ("SECURITY", "Mail gateway", "Cloud", None),
    ("SECURITY", "Mail gateway", "On-Prem", None),
    ("SECURITY", "Mail gateway", None, None),
    ("SECURITY", "MFT", None, None),
    ("SECURITY", "NAC", None, None),
    ("SECURITY", "Other", None, None),
    ("SECURITY", "OTP", None, None),
    ("SECURITY", "PAM", None, None),
    ("SECURITY", "PS", None, None),
    ("SECURITY", "SAAS", None, None),
    ("SECURITY", "SASE", None, None),
    ("SECURITY", "SCADA", None, None),
    ("SECURITY", "SIEM/SOC", None, None),
    ("SECURITY", "SSL VPN", None, None),
    ("SECURITY", "WAF", "Cloud", None),
    ("SECURITY", "WAF", "On-Prem", None),
    ("SECURITY", "WAF", None, None),
    ("SECURITY", "WEB GATEWAY", "Cloud", None),
    ("SECURITY", "WEB GATEWAY", "On-Prem", None),
    ("SECURITY", "WEB GATEWAY", None, None),
    ("SECURITY", None, None, None),
    ("ServiceNow", None, None, None),
    ("TRAINING", "Course", None, None),
    ("TRAINING", "Workshop", None, None),
    ("WORK", "Consulting", None, None),
    ("WORK", "Installation", None, None),
    ("WORK", "Labor", None, None),
    ("WORK", "Project Management", None, None),
)
_VALID_PATH_SET = set(VALID_CATEGORY_PATHS)


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(t in text for t in terms)


def _is_subscription(text: str) -> bool:
    return bool(re.search(r"\b\d+\s*(year|month|yr|mo|years|months)\b", text)) or _contains_any(
        text, ("renewal", "subscription", "subscr", "annual")
    )


def _derive_item_kind(description: str, category: CategoryPath) -> str:
    text = _norm(description)
    software_terms = (
        "license",
        "licence",
        "saas",
        "subscription",
        "renewal",
        "cloud",
        "ms365",
        "os",
        "virtualization",
    )
    service_terms = (
        "support",
        "service contract",
        "consulting",
        "installation",
        "labor",
        "workshop",
        "course",
        "project management",
    )
    if _contains_any(text, service_terms) or category[0] in {"WORK", "TRAINING"}:
        return "service"
    if _contains_any(text, software_terms):
        return "software"
    return "hardware"


def _classify_category_with_reason(
    mpn: str, description: str, hint: Optional[CategoryPath] = None
) -> Tuple[CategoryPath, str]:
    if hint and hint in _VALID_PATH_SET:
        return hint, "hint:validated"

    text = _norm(f"{mpn} {description}")

    if _contains_any(text, ("ins-day", "labor", "consulting", "implementation", "installation")):
        return ("WORK", "Labor", None, None), "rule:work_labor"
    if _contains_any(text, ("course", "workshop", "training")):
        return ("TRAINING", "Course", None, None), "rule:training_course"
    if _contains_any(text, ("transceiver", "qsfp", "gbic", "fg-tran-", "sfp lx", "sfp sx", "sfp rj45", "sfp module")):
        return ("Networking", "Optical", None, None), "rule:networking_optical"
    if _contains_any(text, ("cp-1560", "firewall", "vpn", "checkpoint", "fortigate")):
        if "branch" in text:
            return ("SECURITY", "Firewall", "Branch", None), "rule:security_firewall_branch"
        return ("SECURITY", "Firewall", "Perimeter", None), "rule:security_firewall_perimeter"
    if _contains_any(text, ("cp-har-ep", "endpoint protection", "eps")):
        return ("SECURITY", "EPS", "Traditional", None), "rule:security_eps"
    if _contains_any(text, ("waf",)):
        return ("SECURITY", "WAF", None, None), "rule:security_waf"
    if _contains_any(text, ("siem", "soc")):
        return ("SECURITY", "SIEM/SOC", None, None), "rule:security_siem_soc"
    if _contains_any(text, ("poly", "studio", "vc")):
        return ("Collaboration", "UCC", "VC", None), "rule:collaboration_ucc_vc"
    if _contains_any(text, ("hdmi", "audio", "video", "cable")):
        return ("Collaboration", "AV", "AUDIO VIDEO", None), "rule:collaboration_av"
    if _contains_any(text, ("power supply", "psu", "sp-fg")):
        return (
            ("SECURITY", "Firewall", "License, Support, Accessories", None),
            "rule:security_firewall_accessories",
        )
    if _contains_any(text, ("con-snt", "service contract", "smartnet")):
        return ("Networking", "Enterprise Network", None, None), "rule:networking_service_contract"
    if _contains_any(text, ("switch", "router", "wireless", "network")):
        return ("Networking", "Enterprise Network", "Access", None), "rule:networking_access"
    if _contains_any(text, ("server", "gpu", "storage array")):
        return ("COMPUTERS", "Servers", None, None), "rule:computers_servers"
    if _contains_any(text, ("license", "windows", "os", "virtualization")):
        return ("Computing", "Software", "OS", None), "rule:computing_software_os"

    return ("OEM", None, None, None), "fallback:oem"


def _classify_category(mpn: str, description: str, hint: Optional[CategoryPath] = None) -> CategoryPath:
    return _classify_category_with_reason(mpn=mpn, description=description, hint=hint)[0]


def _path_is_valid(path: CategoryPath) -> bool:
    if path in _VALID_PATH_SET:
        return True

    # Allow valid parent paths when the taxonomy defines only deeper leaves.
    p1, p2, p3, p4 = path
    for cand in _VALID_PATH_SET:
        c1, c2, c3, c4 = cand
        if c1 != p1:
            continue
        if p2 is not None and c2 != p2:
            continue
        if p3 is not None and c3 != p3:
            continue
        if p4 is not None and c4 != p4:
            continue
        return True
    return False


def _build_item_name(manufacturer: str, description: str, mpn: str) -> str:
    base = f"{manufacturer} {description} {mpn}".strip()
    compact = re.sub(r"\s+", " ", base)
    return compact[:50]


MANDATORY_FIELDS: Sequence[str] = (
    "mpn",
    "item_name",
    "item_description",
    "level_1",
    "software_item",
    "inventory_managed",
    "serialized",
    "subscription",
    "manufacturer_code",
    "manufacturer_manager",
    "vendor_code",
    "vendor_part_number",
    "unit_cost",
)


def validate_mandatory_fields(record: CatalogRecord) -> ValidationResult:
    missing: List[str] = []
    for field_name in MANDATORY_FIELDS:
        value = getattr(record, field_name, None)
        if value is None:
            missing.append(field_name)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field_name)
    return ValidationResult(is_valid=not missing, missing_fields=tuple(missing))


def _pick_vendor_from_history(
    history: Sequence[PurchaseHistoryRecord],
    mpn: str,
    manufacturer_name: Optional[str],
    as_of: Optional[date] = None,
) -> Optional[str]:
    if not history:
        return None

    ref = as_of or date.today()
    cutoff = ref - timedelta(days=365)
    matches = [
        h
        for h in history
        if h.purchased_on >= cutoff
        and (
            (h.mpn and _norm(h.mpn) == _norm(mpn))
            or (
                manufacturer_name
                and h.manufacturer_name
                and _norm(h.manufacturer_name) == _norm(manufacturer_name)
            )
        )
    ]
    if not matches:
        return None

    counts: Dict[str, int] = {}
    for record in matches:
        counts[record.vendor_code] = counts.get(record.vendor_code, 0) + 1

    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


class BaseCatalogAgent:
    def __init__(
        self,
        manufacturer_code_lookup: Optional[Dict[str, str]] = None,
        manufacturer_manager_lookup: Optional[Dict[str, str]] = None,
        vendor_code_lookup: Optional[Dict[str, str]] = None,
        erp_client: Optional[ERPClient] = None,
        purchase_history: Sequence[PurchaseHistoryRecord] = (),
        mpn_vendor_search: Optional[Callable[[str], Sequence[str]]] = None,
        web_lookup: Optional[Callable[[str, str], WebCatalogHint]] = None,
    ) -> None:
        self.manufacturer_code_lookup = {_norm(k): v for k, v in (manufacturer_code_lookup or {}).items()}
        self.manufacturer_manager_lookup = {_norm(k): v for k, v in (manufacturer_manager_lookup or {}).items()}
        self.vendor_code_lookup = {_norm(k): v for k, v in (vendor_code_lookup or {}).items()}
        self.erp_client = erp_client or InMemoryERPClient(
            manufacturer_code_lookup=self.manufacturer_code_lookup,
            manufacturer_manager_lookup=self.manufacturer_manager_lookup,
            vendor_code_lookup=self.vendor_code_lookup,
            purchase_history=purchase_history,
        )
        self.mpn_vendor_search = mpn_vendor_search or (lambda _mpn: ())
        self.web_lookup = web_lookup or (lambda _mpn, _manufacturer: WebCatalogHint())

    def _manufacturer_code(self, manufacturer_name: str) -> Optional[str]:
        return self.erp_client.get_manufacturer_code(manufacturer_name)

    def _manufacturer_manager(self, manufacturer_name: str) -> Optional[str]:
        return self.erp_client.get_manufacturer_manager(manufacturer_name)

    def _vendor_code(self, supplier_name: Optional[str]) -> Optional[str]:
        if not supplier_name:
            return None
        return self.erp_client.get_vendor_code(supplier_name)

    def _preferred_vendor_code(self, mpn: str, manufacturer_name: str) -> Optional[str]:
        return self.erp_client.get_preferred_vendor(mpn=mpn, manufacturer_name=manufacturer_name, since_days=365)

    def _apply_hard_gate(
        self,
        record: CatalogRecord,
        *,
        parser_used: str,
        extraction_confidence: str,
        sku_confidence: str,
        message_on_success: str = "Catalog record is ready.",
    ) -> CatalogDecision:
        validation = validate_mandatory_fields(record)
        warnings: List[str] = []
        if extraction_confidence == "low" or sku_confidence == "low":
            warnings.append("Low confidence extraction path was used.")

        trace = {
            "parser_used": parser_used,
            "extraction_confidence": extraction_confidence,
            "sku_confidence": sku_confidence,
            "category_rule": record.category_resolution_reason,
            "erp_manufacturer_code": "hit" if record.manufacturer_code else "miss",
            "erp_manufacturer_manager": "hit" if record.manufacturer_manager else "miss",
            "erp_vendor_code": "hit" if record.vendor_code else "miss",
        }
        if validation.is_valid:
            return CatalogDecision(
                status="ready",
                message=message_on_success,
                record=record,
                warnings=tuple(warnings),
                trace=trace,
            )

        return CatalogDecision(
            status="needs_manual_review",
            message="Missing mandatory fields in catalog record.",
            record=record,
            missing_fields=tuple(validation.missing_fields),
            warnings=tuple(warnings),
            trace=trace,
        )

    def _build_record(
        self,
        mpn: str,
        manufacturer_name: str,
        item_description: str,
        vendor_code: Optional[str],
        unit_cost: Optional[float],
        vendor_part_number: Optional[str],
        web_hint: Optional[WebCatalogHint],
    ) -> CatalogRecord:
        hint_path = web_hint.category_hint if web_hint else None
        category, category_reason = _classify_category_with_reason(
            mpn=mpn, description=item_description, hint=hint_path
        )
        if not _path_is_valid(category):
            category = ("OEM", None, None, None)
            category_reason = "fallback:oem_invalid_path"

        item_kind = _derive_item_kind(item_description, category)
        software_item = item_kind == "software"
        service_item = item_kind == "service"
        inventory_managed = not service_item

        # Serializable only for core physical inventory (hardware and not consumables).
        consumable = _contains_any(_norm(item_description), ("cable", "accessory", "consumable"))
        serialized = inventory_managed and not software_item and not consumable

        subscription = _is_subscription(_norm(item_description))

        return CatalogRecord(
            mpn=mpn,
            item_name=_build_item_name(manufacturer_name, item_description, mpn),
            item_description=item_description,
            level_1=category[0],
            level_2=category[1],
            level_3=category[2],
            level_4=category[3],
            software_item=software_item,
            inventory_managed=inventory_managed,
            serialized=serialized,
            subscription=subscription,
            manufacturer_code=self._manufacturer_code(manufacturer_name),
            manufacturer_manager=self._manufacturer_manager(manufacturer_name),
            vendor_code=vendor_code,
            vendor_part_number=vendor_part_number or mpn,
            unit_cost=unit_cost,
            weight_kg=web_hint.weight_kg if web_hint else None,
            height_cm=web_hint.height_cm if web_hint else None,
            width_cm=web_hint.width_cm if web_hint else None,
            depth_cm=web_hint.depth_cm if web_hint else None,
            datasheet_link=web_hint.datasheet_link if web_hint else None,
            category_resolution_reason=category_reason,
        )


class ManualMpnCatalogAgent(BaseCatalogAgent):
    def __init__(self, *args, require_user_confirmation: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.require_user_confirmation = require_user_confirmation

    def process(
        self,
        request: ManualMpnRequest,
        purchase_history: Sequence[PurchaseHistoryRecord] = (),
    ) -> CatalogDecision:
        mpn = request.mpn.strip()
        if not mpn:
            return CatalogDecision(
                status="needs_manual_review",
                message="MPN is mandatory.",
            )

        manufacturer = request.confirmed_manufacturer_name or request.manufacturer_name
        candidates = tuple(self.mpn_vendor_search(mpn))

        if not manufacturer:
            if not candidates:
                return CatalogDecision(
                    status="needs_manual_review",
                    message="Manufacturer is unknown. Manual manufacturer input required.",
                )
            if len(candidates) > 1 and self.require_user_confirmation:
                return CatalogDecision(
                    status="needs_user_confirmation",
                    message=(
                        f"SKU collision detected for {mpn}. Choose manufacturer candidate."
                    ),
                    vendor_candidates=candidates,
                    trace={"parser_used": request.parser_used},
                )
            if len(candidates) == 1 and self.require_user_confirmation:
                return CatalogDecision(
                    status="needs_user_confirmation",
                    message=f"Identified {mpn} as {candidates[0]}. Please confirm.",
                    vendor_candidates=candidates,
                    trace={"parser_used": request.parser_used},
                )
            manufacturer = candidates[0]

        web_hint = self.web_lookup(mpn, manufacturer)
        merged_description = (web_hint.item_description or "").strip() or f"{manufacturer} {mpn}"
        preferred_vendor = self._preferred_vendor_code(mpn=mpn, manufacturer_name=manufacturer)
        if preferred_vendor is None and purchase_history:
            preferred_vendor = _pick_vendor_from_history(
                purchase_history,
                mpn=mpn,
                manufacturer_name=manufacturer,
            )

        record = self._build_record(
            mpn=mpn,
            manufacturer_name=manufacturer,
            item_description=merged_description,
            vendor_code=preferred_vendor,
            unit_cost=None,
            vendor_part_number=mpn,
            web_hint=web_hint,
        )
        return self._apply_hard_gate(
            record,
            parser_used=request.parser_used,
            extraction_confidence=request.extraction_confidence,
            sku_confidence=request.sku_confidence,
        )


class QuoteParsingCatalogAgent(BaseCatalogAgent):
    def process(self, request: QuoteRequest) -> CatalogDecision:
        mpn = request.mpn.strip()
        if not mpn:
            return CatalogDecision(status="needs_manual_review", message="MPN is mandatory.")

        web_hint = self.web_lookup(mpn, request.manufacturer_name)
        web_kind = web_hint.item_kind
        supplier_kind = _derive_item_kind(request.item_description, ("OEM", None, None, None))

        if web_kind and web_kind != supplier_kind:
            return CatalogDecision(
                status="needs_manual_review",
                message=(
                    "Suspicious supplier mismatch: supplier description conflicts with web identity."
                ),
                trace={
                    "parser_used": request.parser_used,
                    "supplier_kind": supplier_kind,
                    "web_kind": web_kind,
                },
            )

        vendor_code = self._vendor_code(request.supplier_name)
        record = self._build_record(
            mpn=mpn,
            manufacturer_name=request.manufacturer_name,
            item_description=request.item_description,
            vendor_code=vendor_code,
            unit_cost=request.unit_cost,
            vendor_part_number=request.vendor_part_number or request.mpn,
            web_hint=web_hint,
        )
        return self._apply_hard_gate(
            record,
            parser_used=request.parser_used,
            extraction_confidence=request.extraction_confidence,
            sku_confidence=request.sku_confidence,
        )


def process_catalog_envelope(
    envelope: CatalogInputEnvelope,
    extraction_rows: Sequence[CatalogExtractionRow],
    *,
    manual_agent: Optional[ManualMpnCatalogAgent] = None,
    quote_agent: Optional[QuoteParsingCatalogAgent] = None,
) -> List[CatalogDecision]:
    """Orchestrate extract -> enrich -> classify -> validate -> decide."""
    decisions: List[CatalogDecision] = []
    if envelope.source_type == "manual_mpn":
        if not manual_agent:
            raise ValueError("manual_agent is required for manual_mpn source type.")
        for row in extraction_rows:
            decision = manual_agent.process(
                ManualMpnRequest(
                    mpn=row.mpn,
                    manufacturer_name=row.manufacturer_name or None,
                    confirmed_manufacturer_name=row.manufacturer_name or None,
                    parser_used=row.parser_used,
                    extraction_confidence=row.extraction_confidence,
                    sku_confidence=row.sku_confidence,
                )
            )
            decisions.append(decision)
        return decisions

    if not quote_agent:
        raise ValueError("quote_agent is required for non-manual source types.")

    for row in extraction_rows:
        decision = quote_agent.process(
            QuoteRequest(
                mpn=row.mpn,
                item_description=row.item_description,
                manufacturer_name=row.manufacturer_name,
                supplier_name=row.supplier_name or (envelope.supplier_hint or "Unknown Supplier"),
                unit_cost=row.unit_cost,
                parser_used=row.parser_used,
                extraction_confidence=row.extraction_confidence,
                sku_confidence=row.sku_confidence,
            )
        )
        decisions.append(decision)
    return decisions


def parse_history_date(value: str) -> date:
    """Parse ISO date for purchase history integration points."""
    return datetime.strptime(value, "%Y-%m-%d").date()
