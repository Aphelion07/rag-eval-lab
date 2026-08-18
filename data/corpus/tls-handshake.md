# The TLS Handshake
tags: crypto, networking

TLS establishes an authenticated, confidential channel over an untrusted
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
the middle attacks.
