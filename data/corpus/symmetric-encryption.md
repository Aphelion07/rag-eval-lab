# Symmetric Encryption
tags: crypto, fundamentals

Symmetric encryption uses one shared secret key for both encryption and
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
changes.
