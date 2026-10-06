import chess.pgn


def load_pgn(file_path: str):
    """
    Load the first chess game from a PGN file.

    Args:
        file_path: Path to the PGN file.

    Returns:
        chess.pgn.Game object.

    Raises:
        ValueError: If the PGN file contains no game.
    """

    with open(file_path, "r", encoding="utf-8") as pgn_file:
        game = chess.pgn.read_game(pgn_file)

    if game is None:
        raise ValueError("No chess game found in PGN file.")

    return game


def get_game_info(game):
    """
    Extract basic metadata from a chess game.
    """

    return {
        "event": game.headers.get("Event", "Unknown"),
        "white": game.headers.get("White", "Unknown"),
        "black": game.headers.get("Black", "Unknown"),
        "result": game.headers.get("Result", "*"),
        "date": game.headers.get("Date", "Unknown"),
    }


def get_moves(game):
    """
    Return all moves in the game's main line.
    """

    return list(game.mainline_moves())

def get_moves_with_notation(game):
    """
    Extract game moves with move number, ply, color,
    UCI notation, and SAN notation.
    """

    board = game.board()

    moves = []

    for ply, move in enumerate(game.mainline_moves(), start=1):

        san = board.san(move)

        color = "white" if board.turn else "black"

        moves.append({
            "ply": ply,
            "move_number": board.fullmove_number,
            "color": color,
            "uci": move.uci(),
            "san": san,
        })

        board.push(move)

    return moves


if __name__ == "__main__":

    game = load_pgn("data/raw/sample_game.pgn")

    print(get_game_info(game))

    moves = get_moves_with_notation(game)

    for move in moves:
        print(move)