from attic import cloud


class FakeBedrock:
    def __init__(self, text):
        self.text, self.calls = text, []

    def converse(self, **kw):
        self.calls.append(kw)
        return {"output": {"message": {"content": [{"text": self.text}]}}}


def test_bedrock_extraction_is_validated_and_sees_only_redacted_text():
    fake = FakeBedrock('Sure! {"doc_type":"receipt","vendor":"Sharma","purchase_date":"2025-03-14",'
                       '"warranty_months":12,"amount":"36580","model_no":"EVIL","extra":"x"}')
    out = cloud.extract_bedrock("Phone: 9876543210\nGSTIN: 14ABCDE1234F1Z5", client=fake, model_id="m")
    sent = fake.calls[0]["messages"][0]["content"][0]["text"]
    assert "9876543210" not in sent and "14ABCDE1234F1Z5" not in sent
    assert "<document>" in sent and "UNTRUSTED" in fake.calls[0]["system"][0]["text"]
    assert out["warranty_months"]["value"] == 12 and out["amount"]["value"] == 36580.0
    assert "extra" not in out and "expires_on" not in out


def test_bad_model_output_degrades_gracefully():
    assert cloud.extract_bedrock("x", client=FakeBedrock("no json here"), model_id="m") is None

    class Boom:
        def converse(self, **kw):
            raise RuntimeError("throttled")

    assert cloud.extract_bedrock("x", client=Boom(), model_id="m") is None
    assert cloud.summarize_bedrock("x", client=Boom(), model_id="m") is None


def test_disabled_without_model_id(monkeypatch):
    monkeypatch.delenv("ATTIC_BEDROCK_MODEL", raising=False)
    assert cloud.extract_bedrock("x") is None and cloud.summarize_bedrock("x") is None
