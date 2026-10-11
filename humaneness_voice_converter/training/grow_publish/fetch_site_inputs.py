#!/usr/bin/env python3
"""Snapshot the user's S10 listening cases and two fixed target voices.

All 162 MP3s are placed in one TAR on Scratch. No per-sample files are made.
The source page is a temporary Cloudflare URL, so this must run while it is
available; the resulting snapshot freezes UIDs, prompt scores and bytes.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
from urllib.request import urlopen

URL = 'https://adopt-judge-jokes-sea.trycloudflare.com/'
OUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')


def get_json(url: str) -> dict:
    with urlopen(url, timeout=90) as handle:
        return json.load(handle)


def add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mtime = 0
    archive.addfile(member, io.BytesIO(payload))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'site_inputs.tar'
    proof = OUT / 'site_inputs.COMPLETE.json'
    with urlopen(URL, timeout=90) as handle:
        html = handle.read().decode('utf-8')
    match = re.search(r'<script id="listening-data" type="application/json">(.*?)</script>',
                      html, re.S)
    assert match, 'Listening page has no embedded data'
    site = json.loads(match.group(1))
    cases, targets = site['cases'], site['targets']
    assert len(cases) == 160 and len(targets) == 2
    assert {target['label'] for target in targets} == {'A', 'B'}
    assert len({case['uid'] for case in cases}) == 160
    if destination.exists() and proof.exists():
        metadata = OUT / 'site_inputs.json'
        old = json.loads(metadata.read_text())
        assert [c['uid'] for c in old['cases']] == [c['uid'] for c in cases]
        for item, fresh in zip(old['cases'], cases):
            item['license'] = fresh['license']
            item['hf_source'] = fresh['hf_source']
        temporary = metadata.with_suffix('.json.partial')
        temporary.write_text(json.dumps(old, indent=2, ensure_ascii=False) + '\n')
        os.replace(temporary, metadata)
        print(proof.read_text())
        return
    manifest = {'schema': 'chatterbox-s3gen-grow-followup-eval-inputs-v1',
                'source_page': URL, 'cases': [], 'targets': targets,
                'audio_sha256': {}, 'codec': 'MP3 from source listening page'}
    packs: dict[str, dict] = {}
    temporary = destination.with_suffix('.tar.partial')
    with tarfile.open(temporary, 'w') as archive:
        for target in targets:
            name = f'references/{target["label"]}.mp3'
            payload = base64.b64decode(site['audio'][name], validate=True)
            assert len(payload) > 1000
            add(archive, name, payload)
            manifest['audio_sha256'][name] = hashlib.sha256(payload).hexdigest()
        for case in cases:
            original = case['variants']['original']['member']
            pack_url = site['audio_locations'][original]
            if pack_url not in packs:
                packs[pack_url] = get_json(URL + pack_url)
            payload = base64.b64decode(packs[pack_url][original], validate=True)
            assert len(payload) > 1000
            member = f'sources/{case["case_id"]}.mp3'
            add(archive, member, payload)
            manifest['audio_sha256'][member] = hashlib.sha256(payload).hexdigest()
            manifest['cases'].append({key: case[key] for key in
                                      ('uid', 'case_id', 'emotion', 'emotion_index',
                                       'sample', 'lang', 'origin', 'text', 'dur_s',
                                       'scores_raw', 'emotion_score', 'license',
                                       'hf_source',
                                       'eligible_cell_rank')})
    assert len(manifest['audio_sha256']) == 162
    os.replace(temporary, destination)
    manifest['tar_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata = OUT / 'site_inputs.json'
    metadata_tmp = metadata.with_suffix('.json.partial')
    metadata_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    os.replace(metadata_tmp, metadata)
    proof_tmp = proof.with_suffix('.json.partial')
    proof_tmp.write_text(json.dumps({'status': 'COMPLETE', 'cases': 160,
                                     'targets': [t['uid'] for t in targets],
                                     'tar_sha256': manifest['tar_sha256']}, indent=2) + '\n')
    os.replace(proof_tmp, proof)
    print(proof.read_text())


if __name__ == '__main__':
    main()
