# Deadlocks
tags: concurrency, operations

A deadlock occurs when two or more operations each hold a resource the
other needs, so none can proceed. It requires four conditions to hold at once:
mutual exclusion, hold and wait, no preemption, and circular wait.

Breaking any one of the four prevents it. The most practical is eliminating
circular wait by imposing a global order in which locks are acquired, so that
two code paths can never take the same two locks in opposite orders.

Databases detect deadlocks and resolve them by aborting one transaction, which
is why applications must be prepared to retry rather than treat a deadlock as a
fatal error.

Distinct from livelock, where operations keep changing state in response to one
another and still make no progress, and from starvation, where one operation is
perpetually outcompeted.
