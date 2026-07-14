# Vendored T0 extraction core

This is the minimal extraction dependency used by `commit-confluence`. It makes
fresh matrix extraction and the BENCH provenance gates independent of a sibling
`t0-morphology-furnace` checkout.

The eight extraction files named in registered profile provenance are copied
byte-for-byte from the exact local seal state; `./confluence doctor` verifies
their recorded SHA-256 values. The four additional Python files are their
transitive local import closure. The two sealed JSONL files are exclusion-only
inputs used by the fresh-data gates.

The code remains covered by the repository's MIT license. Do not edit these
files in place: changing a stamped file invalidates profile/resume provenance.
