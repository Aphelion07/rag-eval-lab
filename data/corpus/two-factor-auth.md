# Enabling 2FA
tags: identity, howto

2FA adds a second proof of identity beyond the password, so that a stolen
password alone is not enough to sign in.

To enable it on a typical service, open account settings, find the security
section, and choose an authenticator app rather than SMS. Scan the QR code with
the app, enter the six digit code it generates to confirm the pairing, and
store the recovery codes somewhere that is not the same device.

SMS delivery is the weakest common option, since it is vulnerable to SIM
swapping and to interception. Authenticator apps generating time based one time
passwords are markedly better. Hardware security keys using WebAuthn are better
still, because the cryptographic challenge is bound to the site's origin and
therefore cannot be phished.

Disabling 2FA typically requires an existing second factor, by design.
