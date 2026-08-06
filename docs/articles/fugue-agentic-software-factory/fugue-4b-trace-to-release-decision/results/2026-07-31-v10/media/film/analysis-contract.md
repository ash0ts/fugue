# Analysis contract — V10 diagnosis coda

- **Audience:** Maintainers turning a reviewed Agent failure into a governed evaluation.
- **Question:** Where does completed diagnosis stop and the unrun repair begin?
- **Takeaway:** The V10 failure is reviewed evidence; the Skill × MCP intervention is still a proposed, human-gated Study.
- **Out of scope:** Any claim that the proposed repair improves behavior.

| Scene | Time | Relationship | Evidence |
| --- | ---: | --- | --- |
| observed-failure | 0–12s | V10 result → exact-history regression | `article.md#results-appendix-diagnosis-only` |
| reviewed-lock | 12–24s | trace + audit + task + coverage | `article.md#qualification-checklist` |
| authority-boundary | 24–36s | diagnosis → proposal → human approval | `article.md#the-approval-card` |
| unrun-repair | 36–48s | preparation evidence ≠ behavioral result | `article.md#results-appendix-diagnosis-only` |

Green denotes completed reviewed evidence, blue locked identity, violet evidence
interpretation, amber unrun or approval-bound work, and coral failure. The film
is silent, deterministic, and respects the 100 px control-safe area.
