# Security Logging and Monitoring
tags: operations, detection

Logging that nobody reads is a compliance artefact rather than a control.
Useful security logging records authentication attempts, authorisation
failures, administrative actions and data access, with enough context to
reconstruct a sequence of events.

Logs must be tamper resistant. An attacker with write access to the logs on the
host they compromised can erase their trail, so forwarding to a separate system
with append only storage is what makes them evidence.

Alerting is where most programmes fail. Too many alerts and analysts stop
reading them; too few and the detection never fires. Alert on patterns rather
than individual events: a single failed login is noise, five hundred across
distinct accounts from one address is credential stuffing.

Never log credentials, tokens or full card numbers.
