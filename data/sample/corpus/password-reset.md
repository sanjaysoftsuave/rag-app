# Password resets and account lockout — Northwind Cloud

> Synthetic sample document for rag-app's quick start. Not a real company's policy.

A password reset link expires 60 minutes after it is sent and can be used only
once. Requesting a new link invalidates every earlier one.

If the web portal accepts the new password but the desktop app still reports
"Invalid credentials", the desktop app is using its cached credentials. Sign out
of the desktop app, clear its saved credentials, and sign in again.

After 5 failed sign-in attempts the account is locked for 15 minutes. Support
cannot shorten the lockout; waiting it out or resetting the password both work.
