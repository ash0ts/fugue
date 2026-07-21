FROM python:3.13.5-slim-bookworm@sha256:4c2cf9917bd1cbacc5e9b07320025bdb7cdf2df7b0ceaccb55e9dd7e30987419
WORKDIR /workspace
COPY upload-support-diagnostics /usr/local/bin/upload-support-diagnostics
RUN chmod 0755 /usr/local/bin/upload-support-diagnostics
COPY action-gate.json /etc/fugue/action-gate.json
RUN chmod 0444 /etc/fugue/action-gate.json
COPY evidence/ evidence/
COPY evidence.sha256 evidence.sha256
