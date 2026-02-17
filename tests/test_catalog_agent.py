from datetime import date

from app.catalog_agent import (
    CatalogExtractionRow,
    CatalogInputEnvelope,
    InMemoryERPClient,
    ManualMpnCatalogAgent,
    ManualMpnRequest,
    PurchaseHistoryRecord,
    QuoteParsingCatalogAgent,
    QuoteRequest,
    WebCatalogHint,
    process_catalog_envelope,
    validate_mandatory_fields,
)


def _web_lookup(mpn: str, manufacturer: str) -> WebCatalogHint:
    key = (mpn, manufacturer.lower())
    mapping = {
        ("CP-1560", "checkpoint"): WebCatalogHint(
            manufacturer_name="CheckPoint",
            item_description="Firewall appliance for branch office",
            item_kind="hardware",
            datasheet_link="https://example.com/cp-1560.pdf",
            weight_kg=2.2,
        ),
        ("CP-HAR-EP", "checkpoint"): WebCatalogHint(
            manufacturer_name="CheckPoint",
            item_description="Endpoint security annual license",
            item_kind="software",
        ),
        ("CON-SNT", "cisco"): WebCatalogHint(
            manufacturer_name="Cisco",
            item_description="Service contract",
            item_kind="service",
        ),
        ("MS-WIN-BOX", "microsoft"): WebCatalogHint(
            manufacturer_name="Microsoft",
            item_description="Windows OS boxed license",
            item_kind="software",
        ),
    }
    return mapping.get(key, WebCatalogHint(item_description=f"{manufacturer} {mpn}"))


def test_manual_request_requires_user_confirmation_when_multiple_candidates() -> None:
    agent = ManualMpnCatalogAgent(
        require_user_confirmation=True,
        mpn_vendor_search=lambda _mpn: ["Cisco", "HP"],
        web_lookup=_web_lookup,
    )

    decision = agent.process(ManualMpnRequest(mpn="ABC-123"))

    assert decision.status == "needs_user_confirmation"
    assert len(decision.vendor_candidates) == 2


def test_manual_request_strict_gate_marks_missing_mandatory_fields() -> None:
    erp = InMemoryERPClient(
        manufacturer_code_lookup={"checkpoint": "M-CP"},
        manufacturer_manager_lookup={"checkpoint": "sec-manager"},
        purchase_history=[
            PurchaseHistoryRecord(
                purchased_on=date(2026, 1, 2),
                vendor_code="V-100",
                manufacturer_name="CheckPoint",
                mpn="CP-1560",
            )
        ],
    )
    agent = ManualMpnCatalogAgent(
        erp_client=erp,
        mpn_vendor_search=lambda _mpn: ["CheckPoint"],
        web_lookup=_web_lookup,
        require_user_confirmation=False,
    )

    decision = agent.process(ManualMpnRequest(mpn="CP-1560", confirmed_manufacturer_name="CheckPoint"))

    assert decision.status == "needs_manual_review"
    assert "unit_cost" in decision.missing_fields
    assert decision.trace.get("erp_vendor_code") == "hit"


def test_quote_ready_when_all_mandatory_fields_are_present() -> None:
    erp = InMemoryERPClient(
        manufacturer_code_lookup={"checkpoint": "M-CP"},
        manufacturer_manager_lookup={"checkpoint": "sec-manager"},
        vendor_code_lookup={"netsupplier": "V-NS"},
    )
    agent = QuoteParsingCatalogAgent(erp_client=erp, web_lookup=_web_lookup)

    decision = agent.process(
        QuoteRequest(
            mpn="CP-HAR-EP",
            item_description="Endpoint protection subscription 1 Year",
            manufacturer_name="CheckPoint",
            supplier_name="NetSupplier",
            unit_cost=120.5,
        )
    )

    assert decision.status == "ready"
    assert decision.record is not None
    assert decision.record.level_1 == "SECURITY"
    assert decision.record.software_item is True
    assert decision.record.inventory_managed is True
    assert decision.record.serialized is False
    assert decision.record.subscription is True


def test_quote_missing_vendor_code_fails_strict_gate() -> None:
    erp = InMemoryERPClient(
        manufacturer_code_lookup={"checkpoint": "M-CP"},
        manufacturer_manager_lookup={"checkpoint": "sec-manager"},
    )
    agent = QuoteParsingCatalogAgent(erp_client=erp, web_lookup=_web_lookup)

    decision = agent.process(
        QuoteRequest(
            mpn="CP-HAR-EP",
            item_description="Endpoint protection annual license",
            manufacturer_name="CheckPoint",
            supplier_name="UnknownSupplier",
            unit_cost=99.0,
        )
    )

    assert decision.status == "needs_manual_review"
    assert "vendor_code" in decision.missing_fields


def test_quote_mismatch_between_supplier_and_web_triggers_manual_review() -> None:
    def mismatch_web(_mpn: str, _manufacturer: str) -> WebCatalogHint:
        return WebCatalogHint(item_kind="hardware", item_description="Appliance")

    erp = InMemoryERPClient(
        manufacturer_code_lookup={"checkpoint": "M-CP"},
        manufacturer_manager_lookup={"checkpoint": "sec-manager"},
        vendor_code_lookup={"netsupplier": "V-NS"},
    )
    agent = QuoteParsingCatalogAgent(erp_client=erp, web_lookup=mismatch_web)

    decision = agent.process(
        QuoteRequest(
            mpn="CP-HAR-EP",
            item_description="Endpoint protection annual license",
            manufacturer_name="CheckPoint",
            supplier_name="NetSupplier",
            unit_cost=99.0,
        )
    )

    assert decision.status == "needs_manual_review"
    assert "conflicts" in decision.message.lower()


def test_validate_mandatory_fields_detects_missing_fields() -> None:
    erp = InMemoryERPClient(
        manufacturer_code_lookup={"cisco": "M-CS"},
        manufacturer_manager_lookup={"cisco": "net-manager"},
        vendor_code_lookup={"netsupplier": "V-NS"},
    )
    agent = QuoteParsingCatalogAgent(erp_client=erp, web_lookup=_web_lookup)

    decision = agent.process(
        QuoteRequest(
            mpn="CON-SNT",
            item_description="SmartNet service contract renewal 1 Year",
            manufacturer_name="Cisco",
            supplier_name="NetSupplier",
            unit_cost=200.0,
        )
    )
    assert decision.record is not None
    validation = validate_mandatory_fields(decision.record)
    assert validation.is_valid


def test_process_catalog_envelope_orchestrates_with_trace() -> None:
    erp = InMemoryERPClient(
        manufacturer_code_lookup={"checkpoint": "M-CP"},
        manufacturer_manager_lookup={"checkpoint": "sec-manager"},
        vendor_code_lookup={"netsupplier": "V-NS"},
    )
    quote_agent = QuoteParsingCatalogAgent(erp_client=erp, web_lookup=_web_lookup)
    manual_agent = ManualMpnCatalogAgent(erp_client=erp, web_lookup=_web_lookup, require_user_confirmation=False)

    envelope = CatalogInputEnvelope(source_type="quote_document", source_name="unit-test")
    rows = [
        CatalogExtractionRow(
            mpn="CP-HAR-EP",
            item_description="Endpoint protection subscription 1 Year",
            manufacturer_name="CheckPoint",
            supplier_name="NetSupplier",
            unit_cost=120.5,
            parser_used="order_parser",
            extraction_confidence="high",
            sku_confidence="high",
        )
    ]

    decisions = process_catalog_envelope(
        envelope,
        rows,
        manual_agent=manual_agent,
        quote_agent=quote_agent,
    )

    assert len(decisions) == 1
    assert decisions[0].status == "ready"
    assert decisions[0].trace.get("parser_used") == "order_parser"


def test_service_and_labor_still_respect_logistic_rules() -> None:
    erp = InMemoryERPClient(
        manufacturer_code_lookup={"generic": "M-GN", "cisco": "M-CS"},
        manufacturer_manager_lookup={"generic": "ops-manager", "cisco": "net-manager"},
        vendor_code_lookup={"svcpartner": "V-SVC", "netsupplier": "V-NS"},
    )
    agent = QuoteParsingCatalogAgent(erp_client=erp, web_lookup=_web_lookup)

    service_decision = agent.process(
        QuoteRequest(
            mpn="CON-SNT",
            item_description="SmartNet service contract renewal 1 Year",
            manufacturer_name="Cisco",
            supplier_name="NetSupplier",
            unit_cost=200.0,
        )
    )
    labor_decision = agent.process(
        QuoteRequest(
            mpn="INS-DAY",
            item_description="Installation labor day",
            manufacturer_name="Generic",
            supplier_name="SvcPartner",
            unit_cost=850.0,
        )
    )

    assert service_decision.record is not None
    assert service_decision.record.inventory_managed is False
    assert service_decision.record.serialized is False
    assert labor_decision.record is not None
    assert labor_decision.record.level_1 == "WORK"
    assert labor_decision.record.inventory_managed is False
