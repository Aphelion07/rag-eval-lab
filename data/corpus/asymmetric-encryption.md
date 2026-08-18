# Asymmetric Encryption
tags: crypto, fundamentals

Asymmetric encryption, also called public key cryptography, uses a key
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
offers roughly the security of a 3072 bit RSA key.
