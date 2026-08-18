# Rotating API Keys
tags: operations, access

API key rotation replaces credentials on a schedule so that an
undiscovered leak has a bounded lifetime.

Rotation without downtime requires overlap: issue the new key, deploy it
alongside the old one, verify the new key is serving traffic, then revoke the
old one. Systems that permit only one active key at a time force a choice
between an outage and skipping rotation, and the outage usually loses.

Keys should be scoped to the narrowest permission set that works and be
attributable to a single consumer. A shared key used by six services cannot be
revoked without breaking six things, so it never gets revoked.

Detection matters as much as rotation. Secret scanning in repositories and
alerting on use from unexpected addresses catch leaks that a schedule alone
will not.
