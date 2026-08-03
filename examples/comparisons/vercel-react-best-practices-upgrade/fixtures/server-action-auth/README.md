# Server Action authorization fixture

This deliberately incomplete Next.js-style Server Action authenticates the
caller but does not authorize team membership before mutation. The built-in
Node test suite is expected to fail until the action enforces authorization at
the mutation boundary.
