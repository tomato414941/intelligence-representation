from __future__ import annotations

from intrep.gpt_training import GPTTrainingConfig, train_mixed_gpt
from intrep.mixed_corpus import default_mixed_documents
from intrep.mixed_corpus_evaluation import evaluate_mixed_corpus_pairing


def main() -> None:
    documents = default_mixed_documents()
    coverage = evaluate_mixed_corpus_pairing(documents)
    result = train_mixed_gpt(
        documents=documents,
        training_config=GPTTrainingConfig(max_steps=5),
    )

    print("intrep mixed-gpt demo")
    print(
        "corpus:"
        f" documents={len(documents)}"
        f" environment_symbolic={coverage.environment_symbolic_count}"
        f" environment_natural={coverage.environment_natural_count}"
        f" paired_episodes={len(coverage.paired_episode_ids)}"
    )
    print(
        "training:"
        f" tokens={result.token_count}"
        f" steps={result.steps}"
        f" initial_loss={result.initial_loss:.4f}"
        f" final_loss={result.final_loss:.4f}"
        f" best_loss={result.best_loss:.4f}"
    )


if __name__ == "__main__":
    main()
