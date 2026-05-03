from __future__ import annotations

import argparse
from pathlib import Path

import shogi
import torch
from torch import nn

from intrep.shogi_move_choice_checkpoint import load_shogi_move_choice_checkpoint
from intrep.shogi_move_encoding import shogi_candidate_move_features
from intrep.shogi_position_encoding import shogi_position_token_ids_from_sfen


def choose_shogi_move(model: nn.Module, board: shogi.Board, *, device: str = "cpu") -> str:
    scored_moves = score_shogi_legal_moves(model, board, device=device)
    return max(scored_moves, key=lambda item: item[1])[0]


def score_shogi_legal_moves(model: nn.Module, board: shogi.Board, *, device: str = "cpu") -> list[tuple[str, float]]:
    legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
    if not legal_moves:
        raise ValueError("board has no legal moves")
    torch_device = torch.device(device)
    position_token_ids = shogi_position_token_ids_from_sfen(board.sfen()).unsqueeze(0).to(torch_device)
    candidate_move_features = shogi_candidate_move_features(
        legal_moves,
        max_choice_count=len(legal_moves),
    ).unsqueeze(0).to(torch_device)
    candidate_mask = torch.ones((1, len(legal_moves)), dtype=torch.bool, device=torch_device)
    model.eval()
    with torch.no_grad():
        logits = model(position_token_ids, candidate_move_features, candidate_mask).squeeze(0)
    return [(move, float(score)) for move, score in zip(legal_moves, logits.tolist(), strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Play shogi against a trained move-choice checkpoint.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--human-color", choices=("black", "white", "none"), default="black")
    parser.add_argument("--max-plies", type=int, default=256)
    args = parser.parse_args()

    model = load_shogi_move_choice_checkpoint(args.checkpoint_path, device=args.device)
    board = shogi.Board()
    human_turn = {
        "black": shogi.BLACK,
        "white": shogi.WHITE,
        "none": None,
    }[args.human_color]

    for ply in range(args.max_plies):
        if board.is_game_over():
            break
        print(board)
        print(f"sfen: {board.sfen()}")
        if human_turn is not None and board.turn == human_turn:
            move_usi = _read_human_move(board)
        else:
            move_usi = choose_shogi_move(model, board, device=args.device)
            print(f"model: {move_usi}")
        board.push_usi(move_usi)
        print(f"ply {ply + 1}: {move_usi}")
    print(board)
    print(f"final_sfen: {board.sfen()}")


def _read_human_move(board: shogi.Board) -> str:
    legal_moves = {move.usi() for move in board.legal_moves}
    while True:
        move_usi = input("your move (USI): ").strip()
        if move_usi in legal_moves:
            return move_usi
        print("illegal move")


if __name__ == "__main__":
    main()
