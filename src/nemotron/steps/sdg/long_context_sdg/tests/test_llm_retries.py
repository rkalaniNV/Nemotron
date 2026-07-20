from types import SimpleNamespace

import long_context_sdg.llm as llm


class FlakyFacade:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def completion(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("temporary transport failure")
        return SimpleNamespace(message=SimpleNamespace(role="assistant", content="OK"))


def test_call_llm_retries_intermittent_transport_failures(monkeypatch):
    facade = FlakyFacade(failures=7)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    result = llm.call_llm({"assistant": facade}, "assistant", [{"role": "user", "content": "hello"}])
    assert result["content"] == "OK"
    assert facade.calls == 8


def test_extract_json_accepts_first_complete_object_with_trailing_output():
    assert llm._extract_json('Preface {"content":"answer"}\n{"content":"duplicate"}') == {"content": "answer"}


def test_extract_json_accepts_fenced_object_with_trailing_commentary():
    assert llm._extract_json('```json\n{"content":"answer"}\n```\nDone') == {"content": "answer"}
