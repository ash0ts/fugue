from __future__ import annotations

import os

from openai import OpenAI

base_url = os.environ.get("FUGUE_COMPAT_BASE_URL", "http://127.0.0.1:18765")
client = OpenAI(
    api_key="fugue-compatibility-key",
    base_url=f"{base_url.rstrip('/')}/v1",
)

response = client.responses.create(model="fugue-candidate", input="hello")
assert response.output_text == "Fugue compatibility response"

events = list(
    client.responses.create(
        model="fugue-candidate", input="hello", stream=True
    )
)
assert sum(event.type == "response.output_text.delta" for event in events) == 1
assert events[-1].type == "response.completed"

chat = client.chat.completions.create(
    model="fugue-candidate",
    messages=[{"role": "user", "content": "hello"}],
)
assert chat.choices[0].message.content == "Fugue compatibility response"

chunks = list(
    client.chat.completions.create(
        model="fugue-candidate",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
)
assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == (
    "Fugue compatibility response"
)
