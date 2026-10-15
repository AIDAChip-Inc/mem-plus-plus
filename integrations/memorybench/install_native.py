"""Install the Mem++ off-policy plugin in an explicit pinned MemoryBench checkout."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

UPSTREAM = '247ba8c1e327fa297c0d57aaad0e3cbd66bac656'


def install(root):
    root = Path(root).resolve()
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != UPSTREAM:
        raise ValueError('Use the documented pinned MemoryBench checkout')
    here = Path(__file__).resolve().parent
    copies = {'mempp.py': 'src/agent/mempp.py', 'mempp_store.py': 'src/agent/mempp_store.py',
              'mempp_solver.py': 'src/solver/mempp.py'}
    registry = root / 'src/memory_systems.py'
    marker = '# Mem++ optional off-policy integration'
    config = root / 'configs/memory_systems/mempp.json'
    provider = root / 'src/llms/anthropic.py'
    provider_text = provider.read_text()
    marker_line = '        if system:\n            params["system"] = system'
    if provider_text.count(marker_line) != 1 or provider_text.count('            rest.append(m)') != 1:
        raise ValueError('Unexpected upstream Anthropic provider; inspect before installation')
    if any((root/target).exists() for target in copies.values()) or marker in registry.read_text() or config.exists():
        raise FileExistsError('Plugin already exists; inspect rather than overwrite it')
    # Check all destinations before making changes.
    for source, target in copies.items():
        shutil.copyfile(here/source, root/target)
    with registry.open('a') as stream:
        stream.write('\n\n' + marker + '''
register(MemorySystemSpec(
    name="mempp", solver_class="src.solver.mempp.MemppSolver",
    config_class="src.solver.mempp.MemppAgentConfig",
    config_file="configs/memory_systems/mempp.json", paper_name="Mem++",
    supports_on_policy=False,
))
''')
    provider.write_text(provider_text.replace('            rest.append(m)', '            rest.append({"role": m["role"], "content": m.get("content", "")})').replace(marker_line, '        if params.get("top_p") is None:\n            params.pop("top_p", None)\n' + marker_line))
    config.write_text(json.dumps({'llm_provider': 'anthropic', 'llm_config': {
        'model': 'claude-sonnet-4-6', 'temperature': 1, 'top_p': None, 'max_tokens': 4096},
        'memory_cache_dir': './memory_cache/mempp', 'retrieve_k': 50}, indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout',type=Path)
    args=parser.parse_args()
    install(args.checkout)
