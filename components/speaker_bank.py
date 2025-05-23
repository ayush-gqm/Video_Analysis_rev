"""SpeakerBank: Dynamic x-vector speaker bank with character mapping and Milvus/Faiss backend support."""

from typing import Dict, List, Optional, Any
import numpy as np

class SpeakerBank:
    """
    Dynamic speaker bank for x-vector embeddings, supporting mapping to characters and Milvus/Faiss backend.
    """
    def __init__(self, backend: str = "faiss", config: Optional[Dict] = None):
        """
        Initialize the SpeakerBank.

        Args:
            backend: 'faiss' or 'milvus'
            config: Optional backend configuration
        """
        self.backend = backend
        self.config = config or {}
        self.speakers = {}  # speaker_id -> embedding
        self.speaker_to_character = {}  # speaker_id -> character_name
        self.character_to_speaker = {}  # character_name -> speaker_id
        # Placeholder for backend index
        self.index = None
        if backend == "faiss":
            import faiss
            self.index = faiss.IndexFlatL2(self.config.get("dim", 256))
            self._id_list = []
        elif backend == "milvus":
            # Placeholder: actual Milvus connection logic would go here
            self.index = None
        else:
            raise ValueError("Unsupported backend: {}".format(backend))

    def add_speaker(self, speaker_id: str, embedding: np.ndarray, character: Optional[str] = None):
        """
        Add a speaker embedding and optional character mapping.
        """
        self.speakers[speaker_id] = embedding
        if character:
            self.speaker_to_character[speaker_id] = character
            self.character_to_speaker[character] = speaker_id
        if self.backend == "faiss":
            self.index.add(embedding.reshape(1, -1).astype(np.float32))
            self._id_list.append(speaker_id)
        # Milvus logic would go here

    def get_speaker(self, speaker_id: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a speaker.
        """
        return self.speakers.get(speaker_id)

    def update_speaker(self, speaker_id: str, embedding: np.ndarray):
        """
        Update the embedding for a speaker.
        """
        self.speakers[speaker_id] = embedding
        # For faiss, would need to rebuild index for update

    def delete_speaker(self, speaker_id: str):
        """
        Delete a speaker from the bank.
        """
        if speaker_id in self.speakers:
            del self.speakers[speaker_id]
        if speaker_id in self.speaker_to_character:
            char = self.speaker_to_character.pop(speaker_id)
            self.character_to_speaker.pop(char, None)
        # For faiss, would need to rebuild index for delete

    def find_nearest(self, embedding: np.ndarray, top_k: int = 1) -> List[str]:
        """
        Find the nearest speakers to the given embedding.
        """
        if self.backend == "faiss" and len(self._id_list) > 0:
            D, I = self.index.search(embedding.reshape(1, -1).astype(np.float32), top_k)
            return [self._id_list[i] for i in I[0] if i < len(self._id_list)]
        # Milvus logic would go here
        return []

    def map_speaker_to_character(self, speaker_id: str, character: str):
        """
        Map a speaker to a character name.
        """
        self.speaker_to_character[speaker_id] = character
        self.character_to_speaker[character] = speaker_id

    def get_character_for_speaker(self, speaker_id: str) -> Optional[str]:
        """
        Get the character name mapped to a speaker.
        """
        return self.speaker_to_character.get(speaker_id)

    def get_speaker_for_character(self, character: str) -> Optional[str]:
        """
        Get the speaker_id mapped to a character name.
        """
        return self.character_to_speaker.get(character) 