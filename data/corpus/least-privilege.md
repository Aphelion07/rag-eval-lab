# Least Privilege
tags: fundamentals, access

The principle of least privilege states that every user, process and
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
roles and are almost never revoked.
