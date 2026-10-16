# ChessMind
A personalized chess weakness detection and explainable training recommendation system that analyzes a player's historical games, identifies recurring mistake patterns, creates a multidimensional weakness profile, and recommends targeted training.

See [Puzzle dataset acquisition and validation](research/puzzle_dataset.md)
for dependency setup, bounded sample acquisition, inspection, and offline tests.

See [Human game dataset acquisition and validation](research/human_game_dataset.md)
for the streaming PGN loader, bounded Lichess sample, provenance, and limitations.

See [Candidate mistake extraction](research/candidate_mistake_extraction.md)
for provisional move selection, reproducible JSONL, and the small Stockfish pilot.

See [Lightweight chess features](research/chess_feature_engineering.md)
for engine-independent board features, conventions, and candidate feature export.

See [Evidence-based mistake proposals](research/evidence_based_mistake_labeling.md)
for conservative tactical evidence, deeper verification, abstention, and pilot review.
