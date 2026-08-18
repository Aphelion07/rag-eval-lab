# Race Conditions
tags: appsec, concurrency

A race condition occurs when the correctness of a computation depends on
the relative timing of concurrent operations. Security relevant races usually
take the time of check to time of use form: a program verifies a condition,
then acts on it, and the condition changes in between.

The classic example is a balance check followed by a withdrawal. Issue many
requests simultaneously and several may pass the check before any of them
commits, withdrawing more than the balance allowed.

Defences make the check and the action atomic: database transactions with
appropriate isolation, conditional updates that fail if the value changed, or
optimistic locking with a version column.

Rate limiting reduces exploitability without fixing the underlying defect.
