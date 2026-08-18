# Cross-Site Scripting
tags: appsec, injection

Cross-site scripting places attacker controlled script into a page viewed
by another user. Stored XSS persists the payload server side; reflected XSS
returns it from the request; DOM based XSS never involves the server at all and
occurs entirely in client side code.

Impact is bounded by what the victim's session can do: reading data, performing
actions, and stealing tokens accessible to JavaScript.

Defence is contextual output encoding. The correct encoding depends on where
the value lands, since HTML body, attribute, URL and JavaScript contexts each
have different rules, and applying the wrong one leaves the hole open. Content
Security Policy limits impact but is not a substitute for encoding.

Marking cookies HttpOnly prevents script from reading them, which blunts token
theft without addressing the injection itself.
