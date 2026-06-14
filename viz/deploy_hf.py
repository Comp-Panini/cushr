"""
Deploy the cuSHR DAG visualizer to a Hugging Face Space (Docker SDK).

One-time setup (you do this):
  1. Create a free account at https://huggingface.co/join
  2. Make a token at https://huggingface.co/settings/tokens
     -> "New token" -> type: **Write** -> copy it.

Run:
  # Windows PowerShell
  $env:HF_TOKEN="hf_xxx"; python deploy_hf.py --name cushr-viz
  # macOS/Linux
  HF_TOKEN=hf_xxx python deploy_hf.py --name cushr-viz

It creates (or updates) the Space and uploads only what the app needs:
  app.py, Dockerfile, requirements.txt, static/, cushr_viz.db, README.md
Then prints your public, shareable URL.
"""
import os
import sys
import argparse

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cushr-viz', help='Space name (no spaces)')
    ap.add_argument('--token', default=os.environ.get('HF_TOKEN'),
                    help='HF write token (or set HF_TOKEN env var)')
    args = ap.parse_args()

    if not args.token:
        sys.exit('No token. Set HF_TOKEN env var or pass --token hf_xxx '
                 '(create one at https://huggingface.co/settings/tokens, type=Write).')

    db = os.path.join(HERE, 'cushr_viz.db')
    if not os.path.exists(db):
        sys.exit('cushr_viz.db not found — run `python build_db.py` first.')

    api = HfApi(token=args.token)
    user = api.whoami()['name']
    repo_id = f'{user}/{args.name}'
    print(f'Deploying to Space: {repo_id}')

    api.create_repo(repo_id=repo_id, repo_type='space', space_sdk='docker',
                    exist_ok=True)

    # Upload the app files (skip dev-only / local junk).
    api.upload_folder(
        repo_id=repo_id,
        repo_type='space',
        folder_path=HERE,
        allow_patterns=['app.py', 'Dockerfile', 'requirements.txt',
                        'static/*', 'cushr_viz.db'],
    )
    # The Space's README.md must carry the HF frontmatter (sdk: docker, app_port).
    api.upload_file(
        path_or_fileobj=os.path.join(HERE, 'hf_space_README.md'),
        path_in_repo='README.md',
        repo_id=repo_id,
        repo_type='space',
    )

    url = f'https://huggingface.co/spaces/{repo_id}'
    print('\nDone. The Space is building (takes ~2-4 min for the first build).')
    print(f'Shareable link: {url}')


if __name__ == '__main__':
    main()
