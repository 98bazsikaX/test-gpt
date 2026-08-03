"""
=============================================================================
 microgpt parancssori név-kiegészítő
=============================================================================

A `microgpt.py` numpy-os modelljét használja: betanítja a modellt (ha még
nincs `model.pkl`), majd egy megadott előtagot kiegészít.

Futtatás:
    uv run python complete_name.py            # az előtagot interaktívan kéri
    uv run python complete_name.py em         # az előtagot argumentumként adod
"""

import sys

import microgpt

# Betanítás (a model.pkl cache nélkül itt mindig újratanul; a lassú rész)
microgpt.train()

# Az előtag forrása: parancssori argumentum, vagy interaktív bevitel
if len(sys.argv) > 1:
    seed = sys.argv[1]
else:
    seed = input("Enter a name prefix: ").strip()

# A modell csak az érvényes (tanulószett-beli) karaktereket ismeri -> szűrjük
seed = "".join(c for c in seed.lower() if c in microgpt.uchars)
if not seed:
    print("(nem érvényes előtag)")
else:
    print(microgpt.generate(seed, temperature=0.5))
