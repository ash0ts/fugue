FROM python:3.12.10-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git ripgrep && rm -rf /var/lib/apt/lists/*
WORKDIR /testbed
RUN git clone https://github.com/ash0ts/fugue.git . && git checkout --detach 96512017842d68add2546a057f0601de3eaf610e
COPY mutation.patch /tmp/fugue-mutation.patch
RUN git apply /tmp/fugue-mutation.patch && rm /tmp/fugue-mutation.patch && rm -rf .git
RUN pip install -e '.[dev]'
