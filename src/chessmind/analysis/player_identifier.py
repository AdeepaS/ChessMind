
def find_player_games(games, player_name):
    """
    Find all games involving the target player.

    Args:
        games: List of chess.pgn.Game objects.
        player_name: Name of the player to identify.

    Returns:
        List of dictionaries containing the game,
        game number, player's color, and opponent.
    """

    if not player_name or not player_name.strip():
        raise ValueError("Player name cannot be empty.")

    target_name = player_name.strip().casefold()

    player_games = []

    for game_number, game in enumerate(games, start=1):

        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")

        white_match = white.strip().casefold() == target_name
        black_match = black.strip().casefold() == target_name

        if white_match and black_match:
            # Ambiguous game: player appears on both sides
            raise ValueError(
                f"Player appears as both White and Black "
                f"in game {game_number}."
            )

        if white_match:
            color = "white"
            opponent = black

        elif black_match:
            color = "black"
            opponent = white

        else:
            continue

        player_games.append({
            "game_number": game_number,
            "game": game,
            "color": color,
            "opponent": opponent,
            "result": game.headers.get("Result", "*"),
        })

    return player_games
