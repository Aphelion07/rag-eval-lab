# TCP
tags: networking, protocols

TCP is a connection oriented transport protocol providing reliable,
ordered delivery of a byte stream.

A connection opens with a three way handshake: the client sends SYN, the server
replies SYN-ACK, the client answers ACK. Sequence numbers track every byte,
lost segments are retransmitted, and a sliding window plus congestion control
adapt the sending rate to the observed capacity of the path.

The guarantees cost latency. Every connection pays a full round trip before any
data moves, and head of line blocking means one lost segment stalls everything
behind it.

TCP runs on port numbers 0 to 65535. Common assignments include 22 for SSH, 80
for HTTP and 443 for HTTPS.
