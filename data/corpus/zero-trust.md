# Zero Trust
tags: architecture, access

Zero trust discards the assumption that location implies trust. Being
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
marketing to the contrary.
