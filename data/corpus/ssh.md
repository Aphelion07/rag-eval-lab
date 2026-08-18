# SSH
tags: networking, access

SSH provides an encrypted channel for remote shell access, file transfer
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
for pivoting.
