#!/usr/bin/env python3
"""Publish packed-source demo MP3s as individually addressable HF dataset files.

No per-sample audio is written to Scratch or Home. The 682 MP3s are streamed
from existing pack/TAR artifacts into small, resumable HF commits. Once audio
exists remotely, the static Space is uploaded from its three small files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from build_demo_space import AUDIO_REPO, SPACE_REPO, STAGE, make


def publish_audio(api: HfApi, audio: dict[str, bytes]) -> None:
    api.create_repo(AUDIO_REPO, repo_type='dataset', exist_ok=True)
    current = set(api.list_repo_files(AUDIO_REPO, repo_type='dataset'))
    if 'README.md' not in current:
        api.upload_file(path_or_fileobj=str(STAGE / 'AUDIO_README.md'),
                        path_in_repo='README.md', repo_id=AUDIO_REPO,
                        repo_type='dataset', commit_message='Document demo audio and provenance')
        current.add('README.md')
    missing = [(path, payload) for path, payload in sorted(audio.items())
               if path not in current]
    print(f'audio repo: existing {len(audio)-len(missing)}/{len(audio)}, '
          f'to upload {len(missing)}', flush=True)
    for start in range(0, len(missing), 20):
        batch = missing[start:start + 20]
        ops = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=payload)
               for path, payload in batch]
        api.create_commit(repo_id=AUDIO_REPO, repo_type='dataset',
                          operations=ops, num_threads=1, commit_message=(
                              f'Add demo MP3s {start+1}-{start+len(batch)} '
                              f'of {len(missing)} missing files'))
        print(f'audio repo: uploaded {start+len(batch)}/{len(missing)} missing', flush=True)
    final = set(api.list_repo_files(AUDIO_REPO, repo_type='dataset'))
    if not all(path in final for path in audio):
        raise RuntimeError('Audio repository is incomplete; will not publish Space')
    print(f'audio repo complete: {len(audio)} MP3s', flush=True)


def publish_space(api: HfApi) -> None:
    api.create_repo(SPACE_REPO, repo_type='space', space_sdk='static', exist_ok=True)
    for name in ('README.md', 'index.html', 'manifest.json'):
        api.upload_file(path_or_fileobj=str(STAGE / name), path_in_repo=name,
                        repo_id=SPACE_REPO, repo_type='space',
                        commit_message=f'Publish standalone converter demo {name}')
    print(f'https://huggingface.co/spaces/{SPACE_REPO}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    audio = make()
    if args.build_only:
        return
    api = HfApi()
    identity = api.whoami()
    if not any(org['name'] == 'laion' for org in identity.get('orgs', [])):
        raise PermissionError('Authenticated identity has no LAION organization access')
    publish_audio(api, audio)
    publish_space(api)


if __name__ == '__main__':
    main()
