# Authentication
tags: identity, fundamentals

Authentication answers the question: who are you? It is the process of
verifying that a party is who it claims to be, before any decision about what
that party may do.

Factors fall into three categories: something you know such as a password,
something you have such as a hardware token or phone, and something you are
such as a fingerprint. Combining categories is what makes multi factor
authentication meaningful; two passwords are still one factor.

Common weaknesses include credential stuffing against reused passwords,
phishing that captures both a password and a one time code, and session token
theft that bypasses the login step entirely.

Authentication produces an identity. It says nothing about permissions, which
is a separate decision made afterwards.
