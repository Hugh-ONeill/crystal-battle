"""Demo-review harness for the PRISM + FRACTURE commentary duo.

capture_feed — subscribe to the overlay feed (commentary_overlay on :8130) and
               write a silent, readable transcript to read at match end.
compare      — interleave the raw-protocol GROUND TRUTH with that transcript,
               beat by beat, to catch commentary hallucinations (a beat claims
               something the protocol never shows) and drops (a real event no
               beat mentioned). Caught every mirror-attribution flip during the
               duo's debugging.
"""
