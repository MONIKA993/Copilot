import importlib.util
from pathlib import Path
MODULE_PATH = Path(__file__).resolve().parents[0] / "promt.py"
spec = importlib.util.spec_from_file_location("promt", MODULE_PATH)
promt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promt)
print('Loaded promt:', hasattr(promt, 'TOOLS'))
print('Tool names:', list(promt.TOOLS.keys()))
# Add an episodic memory and read back
entry = promt.add_episodic_memory('smoke test episode', session_id='smoke_session')
print('Added episode id:', entry.get('id'))
print('Recent episodes:', promt.get_episodic_memory(2))
print('Search:', promt.search_episodic_memory('smoke'))
