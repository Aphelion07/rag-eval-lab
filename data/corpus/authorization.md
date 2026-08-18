# Authorization
tags: identity, fundamentals

Authorization answers a different question from authentication: given
that we know who you are, what are you allowed to do?

The two are routinely conflated, and the confusion causes real vulnerabilities.
An API that checks a valid session but never checks whether that user owns the
requested record has authenticated correctly and authorized not at all. This
class of bug, broken object level authorization, is consistently among the most
commonly exploited in web applications.

Models include role based access control, where permissions attach to roles and
users receive roles, and attribute based access control, where decisions are
computed from properties of the user, resource and context.

Authorization checks belong on the server, at the point where the resource is
accessed. Hiding a button in the user interface is not an access control.
