# Command Injection
tags: appsec, injection

Command injection occurs when untrusted input reaches a shell. Because
shells interpret metacharacters such as semicolons, pipes and backticks, input
that was meant to be an argument can become an additional command.

It is frequently more severe than SQL injection: successful exploitation gives
code execution as the service account, not merely data access.

The reliable fix is to avoid the shell entirely. Passing an argument vector
directly to the operating system, rather than a single string to be parsed,
removes the interpretation step where the vulnerability lives. Where a shell is
unavoidable, allowlist the permitted values rather than trying to filter out
dangerous characters.

Related variants include argument injection, where input cannot add a new
command but can add a flag that changes the behaviour of the intended one.
