---
name: pdf-artifact-workflow
description: Use when a task requires inspecting, extracting, filling, editing, or redacting PDF files and validating the resulting artifact.
---

# PDF artifact workflow

Treat the requested output file as a deliverable that must be both correct and
visually usable. Preserve the original input and work on a copy or create a new
output at the exact path requested by the task.

## Inspect before editing

Structure and text are the default inspection path. Use metadata, extracted
text, form-field definitions, annotations, page boxes, and coordinates before
sending any page image for inspection. Filter reports to the pages and fields
relevant to the task; summarize locally instead of emitting complete object
trees, field inventories, or binary data.

1. List the available files and read the task's required output paths.
2. Use `file` and `pdfinfo` to identify PDF versions, encryption, page count,
   page dimensions, and form metadata.
3. Use `pdftotext -layout` when text extraction is useful. Compare reading
   order with field and annotation geometry and use bounded OCR for scanned
   content when available. Visual comparison is a fallback governed by the
   bounded visual QA rules below, not a prerequisite.
4. For forms, inspect field names, types, current values, and page placement
   with an available PDF library before deciding how to fill them.
5. Keep a short checklist of required edits, required outputs, and information
   that must remain unchanged.

## Choose the right operation

- Fill existing form fields when usable fields are present. Preserve field
  appearance and verify that values remain visible after saving.
- Use coordinate-aware overlays only for non-fillable pages. Match the source
  page size and rotation, and avoid covering labels, borders, or existing text.
- For extraction and comparison tasks, normalize values deliberately while
  preserving identifiers, numeric types, and the requested output ordering.
- For redaction, remove the underlying content with a real redaction operation.
  Drawing an opaque rectangle is not sufficient because hidden text may remain
  extractable.
- Avoid rasterizing every page unless the task requires it. Rasterization can
  destroy searchable text, form fields, links, and accessibility information.

## Bounded visual QA

Use visual inspection only when appearance cannot be established from
structural and textual checks, such as clipping, overlap, or uncertain overlay
placement. Before invoking an image-inspection capability, require its
documented contract to carry images as structured media without serializing
base64, data URLs, raw bytes, or other binary content into textual tool output.
If that guarantee is absent or unclear, treat visual inspection as unavailable
and continue with structural and textual validation.

When the capability gate passes:

- Render only changed or uncertain pages, never the whole document by default.
- Inspect one page or tight crop at a time, at no more than 100 DPI and 1200
  pixels on the longest edge.
- Keep each rendered file at or below 512 KiB; crop or reduce it before
  inspection when necessary.
- Use at most four previews for one task unless the user explicitly requests a
  broader visual review.
- Never print, paste, or otherwise carry encoded image data through commands,
  tool output, or messages.

## Validate the artifact

1. Confirm every requested output exists at the exact path and can be opened.
2. Re-run `pdfinfo` and compare page count, dimensions, and rotation with the
   source when those properties should be preserved.
3. Run `qpdf --check` when available to detect structural corruption.
4. When the bounded visual QA capability gate passes, inspect only changed or
   uncertain pages or crops for clipping, overlap, misplaced text, blank pages,
   broken fonts, and accidental occlusion. Otherwise make the visual-validation
   limitation explicit and rely on the structural and textual checks; do not
   synthesize visual confirmation.
5. Extract text from the output again. Confirm required values are present and
   redacted values are absent from both visible output and extracted text.
6. For form tasks, reopen the output and verify the saved field values. For
   data tasks, parse the final JSON, CSV, or workbook and verify its schema,
   types, ordering, and counts.
7. Leave unrelated source content unchanged and do not create extra deliverable
   files unless they are useful temporary validation artifacts.
