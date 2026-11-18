"""Check course specifications, current documentation links and shell examples.

Uses only the standard library. Historical archive links to local run artifacts
are reported separately; no model, data, GPU or simulator is loaded.
"""
from pathlib import Path
import ast
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check():
    errors = []
    protocol = json.loads((ROOT / 'configs/course/protocol.json').read_text())
    expected = {'sampler': 'DDPM', 'num_train_timesteps': 100,
                'num_inference_steps': 10, 'sweep_allowed': False}
    if protocol.get('diffusion') != expected:
        errors.append('Course diffusion configuration must remain fixed at DDPM/100/10.')
    if protocol['inputs']['history_length'] != 0 or not protocol['backbone']['frozen']:
        errors.append('Main experiments require no numeric history and a frozen backbone.')
    rows = []
    for folder in ('a', 'b', 'eval'):
        path = ROOT / 'configs/course' / folder / 'experiments.json'
        spec = json.loads(path.read_text())
        if not spec.get('specification_only') or not (path.parent / spec['protocol']).is_file():
            errors.append(f'Invalid specification/protocol reference: {path.relative_to(ROOT)}')
        rows.extend(spec['experiments'])
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != {'A0', 'A1', 'A2', 'A3', 'B0', 'B1', 'B2', 'B3', 'B4', 'C0'}:
        errors.append('Experiment IDs are incomplete or duplicated.')
    for row in rows:
        if not 1 <= row['execution_horizon'] <= row['action_horizon']:
            errors.append(f'Invalid H/E for {row["id"]}')
        if row.get('checkpoint_reference') and row['checkpoint_reference'] not in ids:
            errors.append(f'Unknown checkpoint reference for {row["id"]}')
    by_id = {row['id']: row for row in rows}
    if by_id.get('B0', {}).get('checkpoint_reference') != 'A0' or by_id.get('B4', {}).get('checkpoint_reference') != 'B1':
        errors.append('B0 must reuse A0; B4 must reuse B1.')

    documents = [ROOT / 'README.md']
    for directory in ('docs', 'configs/course', 'src/course_heads', 'src/course_sequence',
                      'evaluation/course', 'scripts/course_train', 'tools', 'third_party', 'data'):
        documents.extend((ROOT / directory).rglob('*.md'))
    shell_examples = 0
    missing_historical = []
    for path in sorted(set(documents)):
        text = path.read_text(encoding='utf-8')
        historical = 'archive' in path.relative_to(ROOT).parts and path != ROOT / 'docs/archive/README.md'
        if text.count('```') % 2:
            errors.append(f'Unclosed code fence: {path.relative_to(ROOT)}')
        for link in re.findall(r'\]\(([^)\n]+)\)', text):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', link) or link.startswith('#'):
                continue
            target = (path.parent / link.split('#')[0]).resolve()
            if not target.exists():
                entry = f'{path.relative_to(ROOT)} -> {link}'
                if historical:
                    missing_historical.append(entry)
                else:
                    errors.append('Broken current link: ' + entry)
        if not historical:
            for language, code in re.findall(r'```([^\n]*)\n(.*?)```', text, re.S):
                if language == 'bash':
                    result = subprocess.run(['bash', '-n'], input=code, text=True, capture_output=True)
                    if result.returncode:
                        errors.append(f'Shell syntax in {path.relative_to(ROOT)}: {result.stderr}')
                    shell_examples += 1
                elif language == 'json':
                    try:
                        json.loads(code)
                    except json.JSONDecodeError as exc:
                        errors.append(f'JSON example in {path.relative_to(ROOT)}: {exc}')
    for directory in ('src', 'scripts', 'utils', 'tools', 'tests', 'airsim_plugin', 'evaluation'):
        for path in (ROOT / directory).rglob('*.py'):
            try:
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f'Python syntax: {exc}')
    print(json.dumps({'experiments': len(rows), 'markdown_documents': len(set(documents)),
                      'shell_examples_checked': shell_examples,
                      'missing_historical_local_links': len(missing_historical),
                      'errors': errors}, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(check())
