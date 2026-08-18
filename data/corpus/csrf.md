# Cross-Site Request Forgery
tags: appsec, web

Cross-site request forgery causes a logged in user's browser to send an
unintended request to an application that trusts it. Because browsers attach
cookies automatically, the request carries the victim's session even though the
attacker never sees it.

The attack changes state rather than reading it: transferring funds, changing
an email address, adding a user. It cannot read the response, which is what
distinguishes it from XSS.

Defences are anti forgery tokens tied to the session and verified server side,
and the SameSite cookie attribute, which instructs the browser not to send
cookies on cross site requests. SameSite=Lax is the modern default and blocks
the common form of the attack.

Checking the Referer header is a weak fallback, since it is frequently stripped
by privacy tooling.
