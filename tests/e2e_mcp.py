"""End-to-end check: real MCP client -> live Streamable HTTP server.

Run:  python -m tests.e2e_mcp   (starts its own server on a free port)
"""
import asyncio, json, os, socket, subprocess, sys, tempfile, time

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from tests.samples import INJECTED, RECEIPT

TOKEN = "e2e-token-" + "x" * 24


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sc(res):
    """structured content of a tool result"""
    return res.structured_content if getattr(res, "structured_content", None) else json.loads(res.content[0].text)


async def main():
    port, data = free_port(), tempfile.mkdtemp()
    env = {**os.environ, "ATTIC_TOKEN": TOKEN, "ATTIC_DATA": data, "ATTIC_PORT": str(port),
           "ATTIC_ALLOWED_HOSTS": f"127.0.0.1:{port}"}
    env.pop("ATTIC_BEDROCK_MODEL", None)
    proc = subprocess.Popen([sys.executable, "-m", "attic"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(50):
            try:
                if httpx.get(base + "/healthz").text == "ok":
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        h = {"Authorization": f"Bearer {TOKEN}"}
        # ingest two documents through the HTTP API (as the upload page would)
        for text in (RECEIPT, INJECTED):
            r = httpx.post(base + "/api/ingest", data={"text": text}, headers=h).json()
            httpx.post(base + "/api/confirm", json={"draft_id": r["draft_id"], "fields": r["fields"]}, headers=h).raise_for_status()

        async with httpx.AsyncClient(headers=h) as http:
            async with streamable_http_client(base + "/mcp", http_client=http) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as s:
                    init = await s.initialize()
                    print("protocol:", init.protocol_version, "| server:", init.server_info.name)
                    tools = {t.name: t for t in (await s.list_tools()).tools}
                    print("tools:", sorted(tools))
                    assert tools["get_warranty_status"].annotations.read_only_hint is True

                    r = sc(await s.call_tool("get_warranty_status", {"item": "my fridge"}))
                    print("fridge ->", r["spoken"])
                    assert r["expires_on"] == "2026-03-14" and r["model_no"] == "RT28C3053S8"

                    r = sc(await s.call_tool("list_expiring", {"days": 3650}))
                    print("expiring ->", r["spoken"])

                    r = sc(await s.call_tool("search_documents", {"query": "mixer"}))
                    mixer = r["results"][0]["id"]
                    print("search ->", r["spoken"])

                    # prompt-injection document: text is returned only as flagged untrusted data
                    r = sc(await s.call_tool("explain_document", {"id": mixer}))
                    assert "IGNORE ALL PREVIOUS" in r["untrusted_document_text"] and "untrusted" in r["note"]
                    notes = sc(await s.call_tool("get_document", {"id": mixer}))["notes"]
                    assert "pwned" not in notes
                    print("injection -> returned as untrusted data, no tool calls triggered")

                    res = await s.read_resource("attic://index")
                    print("resource ->", len(json.loads(res.contents[0].text)), "docs")
                    p = await s.get_prompt("monthly_household_review")
                    print("prompt ->", p.messages[0].content.text[:50], "...")
        # unauthenticated MCP call is rejected
        assert httpx.post(base + "/mcp", json={}).status_code == 401
        print("ALL E2E CHECKS PASSED")
    finally:
        proc.terminate()
        err = proc.stderr.read().decode()[-600:]
        if "Traceback" in err:
            print(err)

asyncio.run(main())
