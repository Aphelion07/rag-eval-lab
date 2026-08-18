# The CIA Triad
tags: fundamentals, governance

The CIA triad names the three properties that information security
protects: confidentiality, integrity and availability.

Confidentiality means data is disclosed only to authorised parties. Integrity
means data cannot be modified undetectably. Availability means the system is
usable when it is needed.

The model's value is that the three regularly conflict, and naming them forces
the trade-off into the open. Encrypting a backup improves confidentiality and
threatens availability if the key is lost. Aggressive rate limiting protects
availability against abuse and denies service to legitimate bursts.

Extensions add authenticity and non repudiation, which some frameworks treat as
aspects of integrity rather than separate properties.
