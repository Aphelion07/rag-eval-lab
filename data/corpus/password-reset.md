# Password Reset Flows
tags: identity, appsec

A password reset flow is an authentication bypass that the application
provides deliberately, which is why its design deserves the same scrutiny as
the login itself.

A sound flow issues a single use, time limited, high entropy token, delivers it
out of band, and invalidates it on use or expiry. The response must be
identical whether or not the account exists, or the endpoint becomes a user
enumeration oracle.

Common failures include tokens derived from predictable values, tokens that
remain valid after use, reset links leaked to third parties through the
Referer header, and host header injection that causes the link in the email to
point at an attacker controlled domain.

Completing a reset should invalidate all existing sessions. Otherwise an
attacker who already has a session keeps it.
