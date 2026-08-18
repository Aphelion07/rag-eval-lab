# SQL Injection
tags: appsec, injection

SQL injection occurs when untrusted input is concatenated into a database
query, allowing an attacker to change the structure of the statement rather
than only its data.

The consequences range from bypassing a login check to dumping entire tables to
writing files on the database host. Blind variants, where the response body
never contains the data, remain exploitable through boolean conditions or time
delays.

The fix is parameterised queries, also called prepared statements. The query
structure is sent to the database separately from the values, so input can
never be interpreted as syntax. Escaping input by hand is not equivalent and
fails on edge cases involving character encodings.

Object relational mappers help but do not immunise: most expose a raw query
escape hatch, and string building inside that hatch is just as vulnerable.
