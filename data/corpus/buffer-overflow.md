# Buffer Overflows
tags: appsec, memory

A buffer overflow writes past the end of an allocated region, corrupting
whatever memory follows. On the stack that adjacent memory frequently includes
the saved return address, so an attacker who controls the overflowing data can
redirect execution.

The class exists because languages such as C and C++ perform no bounds checking
by default, and functions like strcpy and gets copy until they find a
terminator rather than until the destination is full.

Mitigations raise the cost without eliminating the bug: stack canaries detect
overwrites before a function returns, non executable stacks prevent running
injected code, and address space layout randomisation makes useful addresses
unpredictable. Return oriented programming was developed specifically to defeat
non executable memory.

Memory safe languages remove the class rather than mitigating it.
