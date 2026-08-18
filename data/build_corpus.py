"""Generate the evaluation corpus and golden set.

The corpus is written out by this script rather than committed by hand so that
the *design constraints* stay visible and checkable:

* **Adjacent topics get separate documents.** Symmetric vs asymmetric
  encryption, authentication vs authorization, TCP vs UDP, SQL vs command
  injection. A corpus of unrelated documents makes every retriever look good -
  the interesting failures only appear when the wrong answer is plausible.
* **Vocabulary is deliberately split.** Some documents say "2FA", the queries
  asking about them say "two-factor authentication". BM25 cannot bridge that;
  dense retrieval can. Other queries hunt exact strings like "port 22", where
  the advantage reverses.
* **Relevance is graded.** 2 = answers the question, 1 = related but not
  sufficient. nDCG uses the grades, the other metrics treat >0 as relevant.

Run from the repository root:

    python data/build_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# doc_id -> (title, tags, body)
DOCUMENTS: dict[str, tuple[str, str, str]] = {
    "symmetric-encryption": (
        "Symmetric Encryption",
        "crypto, fundamentals",
        """Symmetric encryption uses one shared secret key for both encryption and
decryption. AES is the dominant algorithm, typically with 128 or 256 bit keys,
and operates on fixed size blocks.

Its advantage is speed. Symmetric ciphers run orders of magnitude faster than
public key algorithms, which is why bulk data is almost always protected this
way even inside protocols that begin with public key cryptography.

Its weakness is key distribution. Both parties must already share the secret,
and there is no way to establish it over an untrusted channel using symmetric
primitives alone. Every party added to a conversation multiplies the number of
keys that must be managed.

Modern deployments use authenticated modes such as AES-GCM, which provide
integrity alongside confidentiality. Unauthenticated modes like raw CBC allow
an attacker to modify ciphertext in ways that produce predictable plaintext
changes.""",
    ),
    "asymmetric-encryption": (
        "Asymmetric Encryption",
        "crypto, fundamentals",
        """Asymmetric encryption, also called public key cryptography, uses a key
pair: a public key that may be freely distributed and a private key that must
never leave its owner. Data encrypted with one can only be decrypted with the
other.

This solves the key distribution problem that symmetric encryption cannot.
Two parties who have never met can establish a confidential channel over a
hostile network, which is the foundation of TLS, SSH and signed software
updates.

The cost is performance. RSA and elliptic curve operations are far slower than
a block cipher, so real protocols use asymmetric cryptography only to
authenticate the peers and agree on a shared secret, then switch to a symmetric
cipher for the actual data.

Key sizes are not comparable across families. A 256 bit elliptic curve key
offers roughly the security of a 3072 bit RSA key.""",
    ),
    "authentication": (
        "Authentication",
        "identity, fundamentals",
        """Authentication answers the question: who are you? It is the process of
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
is a separate decision made afterwards.""",
    ),
    "authorization": (
        "Authorization",
        "identity, fundamentals",
        """Authorization answers a different question from authentication: given
that we know who you are, what are you allowed to do?

The two are routinely conflated, and the confusion causes real vulnerabilities.
An API that checks a valid session but never checks whether that user owns the
requested record has authenticated correctly and authorized not at all. This
class of bug, broken object level authorization, is consistently among the most
commonly exploited in web applications.

Models include role based access control, where permissions attach to roles and
users receive roles, and attribute based access control, where decisions are
computed from properties of the user, resource and context.

Authorization checks belong on the server, at the point where the resource is
accessed. Hiding a button in the user interface is not an access control.""",
    ),
    "sql-injection": (
        "SQL Injection",
        "appsec, injection",
        """SQL injection occurs when untrusted input is concatenated into a database
query, allowing an attacker to change the structure of the statement rather
than only its data.

The consequences range from bypassing a login check to dumping entire tables to
writing files on the database host. Blind variants, where the response body
never contains the data, remain exploitable through boolean conditions or time
delays.

The fix is parameterised queries, also called prepared statements. The query
structure is sent to the database separately from the values, so input can
never be interpreted as syntax. Escaping input by hand is not equivalent and
fails on edge cases involving character encodings.

Object relational mappers help but do not immunise: most expose a raw query
escape hatch, and string building inside that hatch is just as vulnerable.""",
    ),
    "command-injection": (
        "Command Injection",
        "appsec, injection",
        """Command injection occurs when untrusted input reaches a shell. Because
shells interpret metacharacters such as semicolons, pipes and backticks, input
that was meant to be an argument can become an additional command.

It is frequently more severe than SQL injection: successful exploitation gives
code execution as the service account, not merely data access.

The reliable fix is to avoid the shell entirely. Passing an argument vector
directly to the operating system, rather than a single string to be parsed,
removes the interpretation step where the vulnerability lives. Where a shell is
unavoidable, allowlist the permitted values rather than trying to filter out
dangerous characters.

Related variants include argument injection, where input cannot add a new
command but can add a flag that changes the behaviour of the intended one.""",
    ),
    "xss": (
        "Cross-Site Scripting",
        "appsec, injection",
        """Cross-site scripting places attacker controlled script into a page viewed
by another user. Stored XSS persists the payload server side; reflected XSS
returns it from the request; DOM based XSS never involves the server at all and
occurs entirely in client side code.

Impact is bounded by what the victim's session can do: reading data, performing
actions, and stealing tokens accessible to JavaScript.

Defence is contextual output encoding. The correct encoding depends on where
the value lands, since HTML body, attribute, URL and JavaScript contexts each
have different rules, and applying the wrong one leaves the hole open. Content
Security Policy limits impact but is not a substitute for encoding.

Marking cookies HttpOnly prevents script from reading them, which blunts token
theft without addressing the injection itself.""",
    ),
    "csrf": (
        "Cross-Site Request Forgery",
        "appsec, web",
        """Cross-site request forgery causes a logged in user's browser to send an
unintended request to an application that trusts it. Because browsers attach
cookies automatically, the request carries the victim's session even though the
attacker never sees it.

The attack changes state rather than reading it: transferring funds, changing
an email address, adding a user. It cannot read the response, which is what
distinguishes it from XSS.

Defences are anti forgery tokens tied to the session and verified server side,
and the SameSite cookie attribute, which instructs the browser not to send
cookies on cross site requests. SameSite=Lax is the modern default and blocks
the common form of the attack.

Checking the Referer header is a weak fallback, since it is frequently stripped
by privacy tooling.""",
    ),
    "tcp": (
        "TCP",
        "networking, protocols",
        """TCP is a connection oriented transport protocol providing reliable,
ordered delivery of a byte stream.

A connection opens with a three way handshake: the client sends SYN, the server
replies SYN-ACK, the client answers ACK. Sequence numbers track every byte,
lost segments are retransmitted, and a sliding window plus congestion control
adapt the sending rate to the observed capacity of the path.

The guarantees cost latency. Every connection pays a full round trip before any
data moves, and head of line blocking means one lost segment stalls everything
behind it.

TCP runs on port numbers 0 to 65535. Common assignments include 22 for SSH, 80
for HTTP and 443 for HTTPS.""",
    ),
    "udp": (
        "UDP",
        "networking, protocols",
        """UDP is a connectionless transport protocol. It adds almost nothing to IP:
source and destination ports, a length and a checksum.

There is no handshake, no retransmission, no ordering and no congestion
control. A datagram is sent and may arrive, arrive twice, or arrive out of
order. Anything stricter must be built by the application.

That minimalism is the point. DNS queries fit in one datagram and are cheaper
to retry than to establish a connection for. Real time audio and video prefer a
dropped frame to a late one. QUIC builds its own reliability and congestion
control on top of UDP precisely to escape TCP's head of line blocking.

The lack of a handshake also makes UDP attractive for reflection and
amplification attacks, since source addresses are trivially spoofed.""",
    ),
    "tls-handshake": (
        "The TLS Handshake",
        "crypto, networking",
        """TLS establishes an authenticated, confidential channel over an untrusted
network. The handshake agrees on a cipher suite, authenticates the server and
derives shared key material.

In TLS 1.3 the client sends its supported parameters and a key share
immediately. The server responds with its own key share, its certificate and a
signature proving possession of the private key. Both sides derive the same
secret via Diffie-Hellman without it ever crossing the wire, and the exchange
completes in one round trip.

Forward secrecy follows from the ephemeral key exchange: compromising the
server's long term private key later does not decrypt recorded past sessions.
TLS 1.3 removed the static RSA key exchange that made this possible.

Certificate validation, not encryption, is what actually prevents machine in
the middle attacks.""",
    ),
    "ssh": (
        "SSH",
        "networking, access",
        """SSH provides an encrypted channel for remote shell access, file transfer
and port forwarding. It listens on TCP port 22 by default.

Server authentication uses a host key that the client pins on first connection,
which is why an unexpected host key change produces a loud warning. Client
authentication is usually by public key: the private key stays on the client
and proves possession by signing a challenge.

Password authentication should be disabled on internet facing hosts, as should
direct root login. Key pairs remove the password from the attack surface
entirely, provided the private key is protected by a passphrase or held in an
agent.

Port forwarding tunnels arbitrary TCP connections through the encrypted
session, which is useful for administration and equally useful to an attacker
for pivoting.""",
    ),
    "cia-triad": (
        "The CIA Triad",
        "fundamentals, governance",
        """The CIA triad names the three properties that information security
protects: confidentiality, integrity and availability.

Confidentiality means data is disclosed only to authorised parties. Integrity
means data cannot be modified undetectably. Availability means the system is
usable when it is needed.

The model's value is that the three regularly conflict, and naming them forces
the trade-off into the open. Encrypting a backup improves confidentiality and
threatens availability if the key is lost. Aggressive rate limiting protects
availability against abuse and denies service to legitimate bursts.

Extensions add authenticity and non repudiation, which some frameworks treat as
aspects of integrity rather than separate properties.""",
    ),
    "least-privilege": (
        "Least Privilege",
        "fundamentals, access",
        """The principle of least privilege states that every user, process and
service should hold only the permissions required for its task, and hold them
only for as long as the task takes.

Its purpose is blast radius reduction. Compromise is assumed rather than
prevented, and the question becomes how far an attacker gets from a foothold.
A web service running as root turns one file write into full host compromise; a
service confined to its own account and directory does not.

Practical forms include separate service accounts per application, scoped and
short lived API tokens rather than long lived administrative ones, and just in
time elevation instead of standing administrator rights.

Privilege creep is the usual failure: permissions accumulate as people change
roles and are almost never revoked.""",
    ),
    "defense-in-depth": (
        "Defence in Depth",
        "fundamentals, architecture",
        """Defence in depth layers independent controls so that the failure of any
one does not lead directly to compromise. It assumes every individual control
will eventually fail.

A web application might sit behind network segmentation, a reverse proxy with
rate limiting, input validation, parameterised queries, a least privileged
database account and monitoring that detects anomalous query volume. An
attacker must defeat all of them, and each buys detection time.

The layers must be genuinely independent to help. Three controls that all
depend on the same identity provider are one control with extra steps.

The cost is complexity, which is itself a source of misconfiguration. More
layers are not automatically better security.""",
    ),
    "zero-trust": (
        "Zero Trust",
        "architecture, access",
        """Zero trust discards the assumption that location implies trust. Being
inside the corporate network grants nothing; every request is authenticated and
authorised on its own merits regardless of origin.

The model responds to the collapse of the network perimeter. Cloud services,
remote work and third party integrations mean there is no longer an inside to
be inside of, and flat internal networks turn one compromised laptop into
lateral movement across everything.

In practice it means strong device and user identity on every request, per
request authorisation decisions rather than per session, micro segmentation
instead of a flat internal network, and the assumption that the network is
already hostile.

It is an architectural direction rather than a product, despite extensive
marketing to the contrary.""",
    ),
    "two-factor-auth": (
        "Enabling 2FA",
        "identity, howto",
        """2FA adds a second proof of identity beyond the password, so that a stolen
password alone is not enough to sign in.

To enable it on a typical service, open account settings, find the security
section, and choose an authenticator app rather than SMS. Scan the QR code with
the app, enter the six digit code it generates to confirm the pairing, and
store the recovery codes somewhere that is not the same device.

SMS delivery is the weakest common option, since it is vulnerable to SIM
swapping and to interception. Authenticator apps generating time based one time
passwords are markedly better. Hardware security keys using WebAuthn are better
still, because the cryptographic challenge is bound to the site's origin and
therefore cannot be phished.

Disabling 2FA typically requires an existing second factor, by design.""",
    ),
    "password-reset": (
        "Password Reset Flows",
        "identity, appsec",
        """A password reset flow is an authentication bypass that the application
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
attacker who already has a session keeps it.""",
    ),
    "api-key-rotation": (
        "Rotating API Keys",
        "operations, access",
        """API key rotation replaces credentials on a schedule so that an
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
will not.""",
    ),
    "race-condition": (
        "Race Conditions",
        "appsec, concurrency",
        """A race condition occurs when the correctness of a computation depends on
the relative timing of concurrent operations. Security relevant races usually
take the time of check to time of use form: a program verifies a condition,
then acts on it, and the condition changes in between.

The classic example is a balance check followed by a withdrawal. Issue many
requests simultaneously and several may pass the check before any of them
commits, withdrawing more than the balance allowed.

Defences make the check and the action atomic: database transactions with
appropriate isolation, conditional updates that fail if the value changed, or
optimistic locking with a version column.

Rate limiting reduces exploitability without fixing the underlying defect.""",
    ),
    "deadlock": (
        "Deadlocks",
        "concurrency, operations",
        """A deadlock occurs when two or more operations each hold a resource the
other needs, so none can proceed. It requires four conditions to hold at once:
mutual exclusion, hold and wait, no preemption, and circular wait.

Breaking any one of the four prevents it. The most practical is eliminating
circular wait by imposing a global order in which locks are acquired, so that
two code paths can never take the same two locks in opposite orders.

Databases detect deadlocks and resolve them by aborting one transaction, which
is why applications must be prepared to retry rather than treat a deadlock as a
fatal error.

Distinct from livelock, where operations keep changing state in response to one
another and still make no progress, and from starvation, where one operation is
perpetually outcompeted.""",
    ),
    "buffer-overflow": (
        "Buffer Overflows",
        "appsec, memory",
        """A buffer overflow writes past the end of an allocated region, corrupting
whatever memory follows. On the stack that adjacent memory frequently includes
the saved return address, so an attacker who controls the overflowing data can
redirect execution.

The class exists because languages such as C and C++ perform no bounds checking
by default, and functions like strcpy and gets copy until they find a
terminator rather than until the destination is full.

Mitigations raise the cost without eliminating the bug: stack canaries detect
overwrites before a function returns, non executable stacks prevent running
injected code, and address space layout randomisation makes useful addresses
unpredictable. Return oriented programming was developed specifically to defeat
non executable memory.

Memory safe languages remove the class rather than mitigating it.""",
    ),
    "owasp-top-ten": (
        "OWASP Top 10",
        "appsec, governance",
        """The OWASP Top 10 is a consensus ranking of the most critical web
application security risks, published periodically and widely used as a
baseline in security requirements and training.

The 2021 edition leads with A01 Broken Access Control, which rose to first
place after appearing in more tested applications than any other category. A02
covers Cryptographic Failures, A03 Injection, which now includes cross site
scripting, and A04 Insecure Design, a category added to capture flaws that no
amount of correct implementation can fix.

It is a awareness document, not a standard and not a checklist. Passing all ten
categories does not mean an application is secure, and OWASP says so
explicitly. The Application Security Verification Standard is the appropriate
tool where a testable requirement set is needed.""",
    ),
    "logging-monitoring": (
        "Security Logging and Monitoring",
        "operations, detection",
        """Logging that nobody reads is a compliance artefact rather than a control.
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

Never log credentials, tokens or full card numbers.""",
    ),
}

# query_id -> (text, {doc_id: grade})
#
# Grades: 2 = answers the question, 1 = related but insufficient.
QUERIES: list[tuple[str, str, dict[str, float]]] = [
    # -- vocabulary gap: the query words do not appear in the document --------
    (
        "q01",
        "How do I turn on two-factor authentication?",
        {"two-factor-auth": 2, "authentication": 1},
    ),
    (
        "q02",
        "What is the process for switching off my second login factor?",
        {"two-factor-auth": 2},
    ),
    (
        "q03",
        "How can I make sure a stolen password is not enough to log in?",
        {"two-factor-auth": 2, "authentication": 1},
    ),
    (
        "q04",
        "How do I stop an attacker who guessed my credentials from getting in?",
        {"two-factor-auth": 2, "authentication": 1},
    ),
    # -- exact-string queries: lexical retrieval should win -------------------
    ("q05", "What port does SSH use?", {"ssh": 2, "tcp": 1}),
    ("q06", "Which port is 443 and which is 80?", {"tcp": 2}),
    ("q07", "What does A01 in the OWASP Top 10 cover?", {"owasp-top-ten": 2, "authorization": 1}),
    ("q08", "What is AES-GCM used for?", {"symmetric-encryption": 2}),
    ("q09", "What does strcpy have to do with security?", {"buffer-overflow": 2}),
    ("q10", "What is SameSite on a cookie?", {"csrf": 2}),
    # -- near-miss pairs: the wrong answer is plausible ----------------------
    (
        "q11",
        "What is symmetric encryption?",
        {"symmetric-encryption": 2, "asymmetric-encryption": 1},
    ),
    (
        "q12",
        "What is asymmetric encryption?",
        {"asymmetric-encryption": 2, "symmetric-encryption": 1},
    ),
    (
        "q13",
        "What is the difference between authentication and authorization?",
        {"authentication": 2, "authorization": 2},
    ),
    ("q14", "What is SQL injection?", {"sql-injection": 2, "command-injection": 1}),
    ("q15", "What is command injection?", {"command-injection": 2, "sql-injection": 1}),
    ("q16", "Explain TCP", {"tcp": 2, "udp": 1}),
    ("q17", "Explain UDP", {"udp": 2, "tcp": 1}),
    ("q18", "What is a race condition?", {"race-condition": 2, "deadlock": 1}),
    ("q19", "What is a deadlock?", {"deadlock": 2, "race-condition": 1}),
    ("q20", "What is XSS?", {"xss": 2, "csrf": 1}),
    ("q21", "What is CSRF?", {"csrf": 2, "xss": 1}),
    # -- conceptual queries spanning several documents ------------------------
    ("q22", "What are the three properties information security protects?", {"cia-triad": 2}),
    ("q23", "Why should a service not run as root?", {"least-privilege": 2, "defense-in-depth": 1}),
    (
        "q24",
        "How do I limit the damage when a system is compromised?",
        {"least-privilege": 2, "defense-in-depth": 2, "zero-trust": 1},
    ),
    ("q25", "Why is being on the internal network not enough to be trusted?", {"zero-trust": 2}),
    (
        "q26",
        "How do layered security controls work together?",
        {"defense-in-depth": 2, "zero-trust": 1},
    ),
    # -- how-to queries -------------------------------------------------------
    ("q27", "How do I safely rotate credentials without downtime?", {"api-key-rotation": 2}),
    ("q28", "What makes a password reset link secure?", {"password-reset": 2}),
    ("q29", "How do I stop untrusted input from changing my database query?", {"sql-injection": 2}),
    ("q30", "How do I avoid passing user input to a shell?", {"command-injection": 2}),
    (
        "q31",
        "What should I do about repeated failed logins from one address?",
        {"logging-monitoring": 2, "authentication": 1},
    ),
    # -- protocol and crypto --------------------------------------------------
    (
        "q32",
        "How does TLS establish a secure connection?",
        {"tls-handshake": 2, "asymmetric-encryption": 1},
    ),
    ("q33", "What is forward secrecy?", {"tls-handshake": 2}),
    (
        "q34",
        "Why do protocols use public key crypto and then switch to a block cipher?",
        {"asymmetric-encryption": 2, "symmetric-encryption": 2, "tls-handshake": 1},
    ),
    ("q35", "What happens during a three way handshake?", {"tcp": 2}),
    ("q36", "Why does QUIC run on top of UDP?", {"udp": 2, "tcp": 1}),
    # -- queries whose best answer is not the obvious keyword match ------------
    (
        "q37",
        "An API checks the session but not who owns the record. What is that?",
        {"authorization": 2, "owasp-top-ten": 1},
    ),
    (
        "q38",
        "Why is checking a balance and then withdrawing unsafe under load?",
        {"race-condition": 2},
    ),
    (
        "q39",
        "What stops an attacker from erasing evidence of an intrusion?",
        {"logging-monitoring": 2},
    ),
    (
        "q40",
        "Why does writing past the end of an array let someone run code?",
        {"buffer-overflow": 2},
    ),
]


def main() -> int:
    corpus_dir = HERE / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, (title, tags, body) in DOCUMENTS.items():
        text = f"# {title}\ntags: {tags}\n\n{body.strip()}\n"
        (corpus_dir / f"{doc_id}.md").write_text(text, encoding="utf-8")

    golden = HERE / "golden.jsonl"
    with golden.open("w", encoding="utf-8") as handle:
        for query_id, text, relevance in QUERIES:
            unknown = set(relevance) - set(DOCUMENTS)
            if unknown:
                raise SystemExit(f"{query_id} labels unknown documents: {sorted(unknown)}")
            handle.write(
                json.dumps({"query_id": query_id, "text": text, "relevance": relevance}) + "\n"
            )

    labelled = sum(len(r) for _, _, r in QUERIES)
    print(f"wrote {len(DOCUMENTS)} documents to {corpus_dir}")
    print(f"wrote {len(QUERIES)} queries with {labelled} labels to {golden}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
