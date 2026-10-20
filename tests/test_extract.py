from attic import extract
from tests.samples import RECEIPT


def test_local_extraction_receipt():
    f = extract.extract_local(RECEIPT)
    assert f["purchase_date"]["value"] == "2025-03-14"        # day-first
    assert f["amount"]["value"] == 36580.0 and f["currency"]["value"] == "INR"
    assert f["warranty_months"]["value"] == 12                # labelled warranty wins
    assert f["model_no"]["value"] == "RT28C3053S8"
    assert f["serial_no"]["value"] == "0ABC12345XYZ"
    assert f["product"]["value"].startswith("Samsung Double Door")
    assert f["vendor"]["value"] == "Sharma Electronics"
    assert f["doc_type"]["value"] == "receipt"


def test_subtotal_never_mistaken_for_total():
    f = extract.extract_local("Subtotal: 100\nDate: 01/02/2025")
    assert "amount" not in f


def test_word_numbers_and_unlabelled_date():
    f = extract.extract_local("Bought 5 March 2024\nTwo year warranty card")
    assert f["warranty_months"]["value"] == 24
    assert f["purchase_date"]["value"] == "2024-03-05"
    assert f["purchase_date"]["confidence"] < 0.6


def test_model_without_digit_rejected():
    assert "model_no" not in extract.extract_local("Model: Samsung")


def test_finalize_computes_expiry_and_spans():
    flat, conf, spans = extract.finalize(extract.extract_local(RECEIPT))
    assert flat["expires_on"] == "2026-03-14"
    assert 0.5 < conf <= 1 and "warranty_months" in spans


def test_clean_fields_rejects_garbage():
    out = extract.clean_fields({
        "doc_type": "malware", "purchase_date": "13/13/2025", "amount": "-5",
        "warranty_months": "9999", "vendor": "  A\n\n B  ", "bogus": "x",
    })
    assert out == {"doc_type": "other", "vendor": "A B"}


def test_merge_prefers_higher_confidence_but_not_identifiers():
    local = extract.extract_local(RECEIPT)
    cloud = {
        "vendor": {"value": "Cloud Vendor", "confidence": 0.99, "span": "bedrock"},
        "amount": {"value": 1.0, "confidence": 0.1, "span": "bedrock"},
        "model_no": {"value": "HALLUCINATED", "confidence": 0.99, "span": "bedrock"},
    }
    m = extract.merge(local, cloud)
    assert m["vendor"]["value"] == "Cloud Vendor"
    assert m["amount"]["value"] == 36580.0
    assert m["model_no"]["value"] == "RT28C3053S8"


def test_redaction_removes_identifiers_but_keeps_product_facts():
    r = extract.redact(RECEIPT)
    for leaked in ("9876543210", "14ABCDE1234F1Z5", "Rohan"):
        assert leaked not in r
    assert "Samsung Double Door Refrigerator" in r and "14/03/2025" in r
