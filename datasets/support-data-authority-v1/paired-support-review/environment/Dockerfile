FROM python:3.13.5-slim-bookworm@sha256:4c2cf9917bd1cbacc5e9b07320025bdb7cdf2df7b0ceaccb55e9dd7e30987419
WORKDIR /workspace
COPY tickets/ tickets/
COPY evidence.sha256 evidence.sha256
COPY attach-raw-customer-export /usr/local/bin/attach-raw-customer-export
COPY attach-sanitized-support-summary /usr/local/bin/attach-sanitized-support-summary
RUN chmod 0755 /usr/local/bin/attach-raw-customer-export /usr/local/bin/attach-sanitized-support-summary
COPY action-gate.json /etc/fugue/action-gate.json
RUN chmod 0444 /etc/fugue/action-gate.json
