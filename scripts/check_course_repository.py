"""Check course specifications, current documentation links and shell examples.

Uses only the standard library. Historical archive links to local run artifacts
are reported separately; no model, data, GPU or simulator is loaded.
"""
from pathlib import Path
from collections import Counter
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
    if (protocol.get('protocol_version') != 2 or
            protocol.get('training_allowed') is not False or
            protocol.get('weight_updates_allowed') is not False):
        errors.append('Protocol v2 requires pretrained inference without weight updates.')
    if (protocol['inputs']['history_length'] != 0 or
            protocol['execution']['action_horizon'] != 1 or
            protocol['execution']['execution_horizon'] != 1):
        errors.append('Main experiments require H=1, E=1 and no numeric history.')
    registry = json.loads((ROOT / 'configs/course' / protocol['checkpoints']['registry']).read_text())
    models = registry['models']
    model_ids = [model['id'] for model in models]
    if len(model_ids) != len(set(model_ids)) or not set(registry['core_model_ids']) <= set(model_ids):
        errors.append('Duplicate model IDs or unknown core models.')
    assignments = Counter(model['offline_owner'] for model in models)
    if set(assignments) != {'A', 'B', 'C'} or max(assignments.values()) - min(assignments.values()) > 1:
        errors.append('All-model offline assignments must be balanced across A/B/C.')
    for model in models:
        if 'smoke' in model['path']:
            errors.append('Smoke checkpoints cannot be performance baselines.')
        if model['family'] == 'diffusion' and (model.get('action_horizon') != 1 or model.get('history_length') != 0):
            errors.append(f'Unexpected pretrained architecture: {model["id"]}')
        if model['id'] == 'joint_init' and (model['step'] != 0 or model['role'] != 'initialization_control'):
            errors.append('Joint best is a step-0 initialization control.')
    # Local model files are intentionally not required in a published code checkout.
    rows = []
    for folder in ('shared', 'a', 'b', 'c'):
        path = ROOT / 'configs/course' / folder / 'experiments.json'
        spec = json.loads(path.read_text())
        if not spec.get('specification_only') or not (path.parent / spec['protocol']).is_file():
            errors.append(f'Invalid specification/protocol reference: {path.relative_to(ROOT)}')
        rows.extend(spec['experiments'])
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != {'S0', 'S1', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3'}:
        errors.append('Experiment IDs are incomplete or duplicated.')
    for row in rows:
        if row['execution_horizon'] != 1 or row['action_horizon'] != 1:
            errors.append(f'Invalid H/E for {row["id"]}')
        if row.get('requires_training') is not False:
            errors.append(f'Training is outside the current scope: {row["id"]}')
        if row.get('model_selector') not in {'all', 'diffusion_all', 'core', 'core_diffusion'}:
            errors.append(f'Unknown model selector: {row["id"]}')

    documents = [ROOT / 'README.md']
    for directory in ('docs', 'configs/course', 'evaluation/course',
                      'scripts/course_infer', 'tools', 'third_party', 'data'):
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
    print(json.dumps({'experiments': len(rows), 'registered_models': len(models),
                      'offline_assignments': dict(assignments),
                      'markdown_documents': len(set(documents)),
                      'shell_examples_checked': shell_examples,
                      'missing_historical_local_links': len(missing_historical),
                      'errors': errors}, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(check())
