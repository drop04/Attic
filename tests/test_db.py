from datetime import date

import pytest

from attic.db import Store, days_left, fts_query, spoken_span


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    s.add_document({"product": "Samsung Refrigerator", "vendor": "Sharma Electronics", "doc_type": "receipt",
                    "purchase_date": "2025-03-14", "warranty_months": 12, "expires_on": "2026-03-14"},
                   ocr_text="double door 253L", spans={"warranty_months": "Warranty: 1 year"})
    s.add_document({"product": "Prestige Mixer Grinder", "vendor": "Priya Appliances", "doc_type": "warranty",
                    "expires_on": "2026-11-02"}, ocr_text="mixer 750W")
    s.add_document({"title": "School circular", "doc_type": "notice"}, ocr_text="annual day schedule")
    return s


def test_search_finds_by_product_word_and_ignores_filler(store):
    assert store.search("when does my fridge warranty expire?")[0]["vendor"] == "Sharma Electronics"
    assert store.search("refrigerator")[0]["vendor"] == "Sharma Electronics"
    assert store.search("the samsung")[0]["product"] == "Samsung Refrigerator"
    assert store.search("mixer")[0]["product"] == "Prestige Mixer Grinder"


def test_search_prefix_and_doc_type_filter(store):
    assert store.search("refrig")[0]["product"] == "Samsung Refrigerator"
    assert store.search("sharma", doc_type="warranty") == []


def test_ocr_text_is_searchable_and_hostile_query_is_safe(store):
    assert store.search("253L")[0]["product"] == "Samsung Refrigerator"
    store.search('"; DROP TABLE documents; -- AND OR NOT (')                  # must not raise
    assert store.search("")                                                    # recent docs


def test_expiring_window(store):
    today = date(2026, 10, 21)
    exp = store.expiring(30, today=today)
    assert [d["product"] for d in exp] == ["Prestige Mixer Grinder"]          # fridge expired >30d ago
    assert len(store.expiring(300, today=date(2026, 2, 1))) == 2


def test_notes_become_searchable(store):
    doc = store.search("refrigerator")[0]
    assert store.add_note(doc["id"], "original kept in blue folder")
    assert store.search("blue folder")[0]["id"] == doc["id"]
    assert not store.add_note(9999, "x")


def test_facts_roundtrip_and_drafts(store):
    doc = store.search("refrigerator")[0]
    assert store.get(doc["id"])["spans"]["warranty_months"] == "Warranty: 1 year"
    d = store.create_draft(None, "txt", {"a": 1})
    assert store.pop_draft(d)["fields"] == {"a": 1}
    assert store.pop_draft(d) is None


def test_speakable_spans():
    assert spoken_span(3) == "expires in 3 days"
    assert spoken_span(90) == "expires in about 3 months"
    assert spoken_span(-20) == "expired about 3 weeks ago"
    assert days_left("2026-10-25", date(2026, 10, 21)) == 4
    assert fts_query("the 'fridge'") == '"fridge"* OR "refrigerator"*'
