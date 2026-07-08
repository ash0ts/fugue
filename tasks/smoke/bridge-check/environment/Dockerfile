# Minimal task environment. Harness installers (run by the Harbor agent
# adapters at setup time) apt-get what they need; keep the base lean.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git procps ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN printf '%s\n' \
    '# Ops notes' \
    '' \
    'The gateway listens on port 7431. Do not confuse it with the metrics' \
    'endpoint on 9090, which is scrape-only.' \
    > /app/NOTES.md
