#!/usr/bin/env python3
"""Curate a cleaner, more real-speech-heavy 40-case public VC comparison.

Only selects from the 160 already prepared, licensed inputs. Selection is
independent of converter output metrics. All source audio remains packed.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path('/e/scratch/laionize/schuhmann1')
INPUT = ROOT / 'chatterbox_s3gen_grow_followup_eval'
OUT = ROOT / 'hvc_demo_v2'

# Public demo curation: replace profanity/sexual content, garbled or banal
# utterances, and implausible synthetic lines. Some emotions are represented
# twice rather than retaining a poor source merely to fill every class.
REPLACEMENTS = {
    '01-2': '01-1',  # actual conversational banter, not a one-line synthesis
    '02-4': '02-1',  # expressive German real speech; no English profanity
    '03-3': '03-2',  # fluent astonishment instead of profane clip
    '05-4': '05-1',  # genuine dramatic performance
    '06-4': '06-3',  # genuine explanation rather than assembly-script synthesis
    '08-3': '08-1',  # intelligible contemplation instead of garbled transcript
    '09-4': '09-1',  # genuine German dramatic contempt
    '10-4': '10-2', # coherent German rather than odd German/English mixture
    '11-4': '11-1', # genuine disappointment instead of garbled PDF line
    '12-3': '12-2', # avoid explicit insult/sexual wording
    '15-1': '30-1', # stronger genuine joyful German performance
    '18-4': '18-2', # more expressive exhaustion, less banal shopping copy
    '21-4': '21-2', # cleaner and better-scored enthusiasm
    '22-2': '19-2', # replace incoherent line; preserve expressive German fear
    '25-4': '26-4', # replace unnatural language-mixed driving line
    '28-1': '14-3', # replace semantically incoherent fragment with genuine doubt
    '32-4': '32-1', # strong-quality German real-speech relief
    '34-4': '27-1', # omit sensual material; genuine German poetic longing
    '36-3': '36-4', # omit hostile/profane speech
    '37-4': '37-1', # actual playful teasing rather than finance copy
    '38-4': '38-2', # clear gratitude without a garbled proper name
    '39-1': '35-1', # replace trivial conjugation with genuine shame scene
}


def main() -> None:
    base = json.loads((INPUT / 'space_stage/CURATED.json').read_text())['selected_case_ids']
    inventory = {row['case_id']: row for row in
                 json.loads((INPUT / 'space_stage/INPUTS.json').read_text())['cases']}
    selected = [REPLACEMENTS.get(cid, cid) for cid in base]
    assert len(base) == len(selected) == len(set(selected)) == 40
    assert set(REPLACEMENTS).issubset(base)
    assert all(inventory[cid]['license'] == 'CC-BY-4.0' for cid in selected)
    banned = re.compile(r'\b(?:fuck(?:ing)?|shit|cum|ass|motherfucker)\b', re.I)
    assert all(not banned.search(inventory[cid]['text']) for cid in selected)
    counts = Counter((inventory[cid]['lang'], inventory[cid]['origin']) for cid in selected)
    payload = {
        'schema': 'humaneness-vc-demo-curation-v2',
        'selected_case_ids': selected,
        'replacements': REPLACEMENTS,
        'counts': {f'{lang}_{origin}': n for (lang, origin), n in sorted(counts.items())},
        'distinct_emotions': len({inventory[cid]['emotion'] for cid in selected}),
        'selection_policy': ('Editor-reviewed text and source-quality proxies; choose genuine real '
                             'speech where content, intelligibility and recording quality hold up. '
                             'Replacements never use converter-output scores. One source may represent '
                             'an emotion already present, rather than retaining an unsafe or weak clip.'),
        'source_inventory': str(INPUT / 'space_stage/INPUTS.json'),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'CURATION.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': 'CURATED', 'counts': payload['counts'],
                      'distinct_emotions': payload['distinct_emotions'],
                      'replacements': len(REPLACEMENTS)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
