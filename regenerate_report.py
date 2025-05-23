import json
from pathlib import Path
from main_parallel import generate_markdown_report

# Load the structured analysis
with open('Kuma_ep5/full_structured_analysis.json', 'r') as f:
    results = json.load(f)

# Generate a new report
generate_markdown_report(results, Path('Kuma_ep5'))

print('Report generated at Kuma_ep5/analysis_report.md') 