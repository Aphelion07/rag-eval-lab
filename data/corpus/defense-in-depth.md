# Defence in Depth
tags: fundamentals, architecture

Defence in depth layers independent controls so that the failure of any
one does not lead directly to compromise. It assumes every individual control
will eventually fail.

A web application might sit behind network segmentation, a reverse proxy with
rate limiting, input validation, parameterised queries, a least privileged
database account and monitoring that detects anomalous query volume. An
attacker must defeat all of them, and each buys detection time.

The layers must be genuinely independent to help. Three controls that all
depend on the same identity provider are one control with extra steps.

The cost is complexity, which is itself a source of misconfiguration. More
layers are not automatically better security.
