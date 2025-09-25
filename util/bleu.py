import sacrebleu


def bleu4_score(reference: str, candidate: str, tokenize: str) -> float:
    bleu = sacrebleu.corpus_bleu(
        candidate, [reference], force=True, tokenize=tokenize
    )
    return bleu.score


# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# chencherry = SmoothingFunction()
# def bleu4_score(reference: str, candidate: str) -> float:
#     return sentence_bleu(
#         [reference],
#         candidate,
#         weights=(0.25, 0.25, 0.25, 0.25),
#         smoothing_function=chencherry.method4,
#     )


# pip install "sacrebleu[ja,ko]"
