import torch
from transformers import LogitsProcessor


class InteractiveLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, k=5):
        self.tokenizer = tokenizer
        self.k = k
        self.prompt_length = -1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_index = 0
        if self.prompt_length == -1:
            # 첫 호출시 input_ids의 길이가 프롬프트의 길이
            self.prompt_length = input_ids.shape[1]

        generated_token_ids = input_ids[batch_index, self.prompt_length :]

        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        current_step = len(generated_token_ids) + 1

        logits = scores[batch_index]
        top_k_logits, top_k_indices = torch.topk(logits, k=self.k)

        candidate_tokens = [self.tokenizer.decode(token_id, skip_special_tokens=False) for token_id in top_k_indices]

        print()
        print(f"===== step {current_step} =====")
        print(f"> {generated_text}")

        print("\t< next token candidates >")
        for i in range(self.k):
            print(f"[{i+1}] '{candidate_tokens[i]}' (logit: {top_k_logits[i].item():.4f})")
        print("[0] EOS (exit)")

        chosen = -1
        while chosen < 0 or chosen > self.k:
            try:
                chosen = int(input(f"select (1-{self.k}, 0=exit): "))
            except ValueError:
                print("try again")

        if chosen == 0:
            chosen_token_id = self.tokenizer.eos_token_id
            print(f"-- selected: EOS --")
        else:
            chosen_token_id = top_k_indices[chosen - 1]
            print(f"-- selected: '{candidate_tokens[chosen - 1]}' --")

        new_logits = torch.full_like(logits, -float("inf"))
        new_logits[chosen_token_id] = 0.0

        return new_logits.unsqueeze(0)
