# UDP
tags: networking, protocols

UDP is a connectionless transport protocol. It adds almost nothing to IP:
source and destination ports, a length and a checksum.

There is no handshake, no retransmission, no ordering and no congestion
control. A datagram is sent and may arrive, arrive twice, or arrive out of
order. Anything stricter must be built by the application.

That minimalism is the point. DNS queries fit in one datagram and are cheaper
to retry than to establish a connection for. Real time audio and video prefer a
dropped frame to a late one. QUIC builds its own reliability and congestion
control on top of UDP precisely to escape TCP's head of line blocking.

The lack of a handshake also makes UDP attractive for reflection and
amplification attacks, since source addresses are trivially spoofed.
