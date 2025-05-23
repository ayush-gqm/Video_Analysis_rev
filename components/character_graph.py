"""CharacterGraph: Aggregates and resolves character identities, builds a knowledge graph of character interactions."""

from typing import Dict, List, Optional, Any
import networkx as nx

class CharacterGraph:
    """
    Aggregates character identities and builds a knowledge graph of character co-occurrence and interactions.
    """
    def __init__(self):
        """
        Initialize the CharacterGraph.
        """
        self.G = nx.Graph()
        self.scene_characters = []  # List of sets of character names per scene
        self.character_map = {}  # speaker_id or alias -> canonical character name

    def add_scene(self, scene_idx: int, speakers: List[str], dialogue: List[Dict]):
        """
        Add a scene's character/speaker data to the graph.

        Args:
            scene_idx: Index of the scene
            speakers: List of speaker/character names in the scene
            dialogue: List of dialogue segments (dicts with 'speaker', 'text', etc.)
        """
        char_set = set(speakers)
        self.scene_characters.append(char_set)
        for char in char_set:
            if char not in self.G:
                self.G.add_node(char, scenes=[scene_idx])
            else:
                self.G.nodes[char]["scenes"].append(scene_idx)
        # Add edges for co-occurrence
        for c1 in char_set:
            for c2 in char_set:
                if c1 != c2:
                    if self.G.has_edge(c1, c2):
                        self.G[c1][c2]["weight"] += 1
                        self.G[c1][c2]["scenes"].append(scene_idx)
                    else:
                        self.G.add_edge(c1, c2, weight=1, scenes=[scene_idx])

    def resolve_characters(self, speaker_bank: Any):
        """
        Resolve character identities using the speaker bank (map aliases to canonical names).
        Args:
            speaker_bank: SpeakerBank instance with speaker-to-character mapping
        """
        for node in list(self.G.nodes):
            canonical = speaker_bank.get_character_for_speaker(node)
            if canonical and canonical != node:
                nx.relabel_nodes(self.G, {node: canonical}, copy=False)

    def build_graph(self):
        """
        Finalize the graph (optional, placeholder for future logic).
        """
        pass

    def export_json(self) -> Dict:
        """
        Export the knowledge graph as a JSON-serializable dict.
        """
        data = nx.readwrite.json_graph.node_link_data(self.G)
        return data 