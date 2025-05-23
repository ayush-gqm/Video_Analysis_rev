import json

# Load the gemini data
with open('Kuma_ep5/gemini/scene_analysis.json', 'r') as f:
    data = json.load(f)

# Print the keys for scene 0
print("Keys in scene 0:", list(data['scenes']['0'].keys()))

# Check the description field
desc = data['scenes']['0'].get('description', '')
print('Description length:', len(desc))
print('First 100 chars:', desc[:100] if desc else 'None')

# Check scene 1 too for comparison
print("\nKeys in scene 1:", list(data['scenes']['1'].keys()))
desc = data['scenes']['1'].get('description', '')
print('Description length:', len(desc))
print('First 100 chars:', desc[:100] if desc else 'None') 