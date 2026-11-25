from pathlib import Path
import re

html = Path("static/index.html").read_text(encoding="utf-8")

# Check key strings
checks = [
    'id="stage-gate-modal"',
    'renderGateBadge',
    'openStageGateModal',
    'closeStageGateModal',
    'runBatchStageGate',
    'runSingleStageGate',
    'PIPELINE_AVAILABLE_STAGES',
    'togglePipelineStage',
    'selectAllPipelineStages',
    'renderStageGateModalContent',
    'Gerbang Mutu'
]

for c in checks:
    count = html.count(c)
    print(f"'{c}': found {count} time(s)")

# Basic HTML tag balance check for the modal
modal_match = re.search(r'<div id="stage-gate-modal".*?</div>\s*</div>\s*</div>', html, re.DOTALL)
if modal_match:
    print("Stage gate modal structure detected cleanly.")
else:
    print("Warning: stage gate modal closing tag regex check.")
