"""Start the operations console on http://127.0.0.1:8765/ (Ctrl-C to stop)."""

from aura.host.console import ConsoleServer
from aura.lab import LabConfig, PatchConfig, VirtualLab

lab = VirtualLab(LabConfig(seed=6, patches=[PatchConfig(0, "surface:3"), PatchConfig(1, "rep:5")]))
srv = ConsoleServer(lab, "127.0.0.1", 8765)
print("console: http://%s:%d/" % srv.address[:2])
srv.serve_forever()
