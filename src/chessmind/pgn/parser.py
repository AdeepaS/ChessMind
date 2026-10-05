import chess.pgn


def parse_game(file_path):
    """Read the first chess game from a PGN file."""

    with open(file_path, "r", encoding="utf-8") as pgn_file:
        game = chess.pgn.read_game(pgn_file)

    return game


def print_game_positions(game):
    """Print every move and the resulting FEN position."""

    board = game.board()

    for move_number, move in enumerate(game.mainline_moves(), start=1):

        san_move = board.san(move)
        board.push(move)

        print(f"Move {move_number}: {san_move}")
        print(f"FEN: {board.fen()}")
        print()


if __name__ == "__main__":

    game = parse_game("data/raw/sample_game.pgn")

    print(f"White: {game.headers.get('White')}")
    print(f"Black: {game.headers.get('Black')}")
    print()

    print_game_positions(game)