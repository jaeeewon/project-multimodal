import os, soundfile as sf
from tts import TTS
from datasets import load_dataset
from tqdm import tqdm

slp_dir = "/home/jpong/Workspace/jaeeewon/slurp/wav"


def get_slurp_sf(skip_exist=False, write_if_ne=False):
    ds = load_dataset("marcel-gohsen/slurp", split="test")
    slp: list[dict] = []  # {id, path, slots, transcript}

    for idx, rec in tqdm(
        enumerate(ds)
    ):  # duplicated id -> index required to distinguish
        if not rec["slots:"]:
            continue  # skip empty 'slots:'
        path = os.path.join(slp_dir, f"{idx:05d}_{rec['id']:05d}.wav")
        if not skip_exist:
            exists = os.path.exists(path)
            if not write_if_ne:
                assert exists, f"not exists: {path}"

            if not exists:
                sf.write(path, rec["audio"]["array"], rec["audio"]["sampling_rate"])
        slots = "; ".join(rec["slots:"])
        slp.append(
            {
                "idx": idx,
                "id": rec["id"],
                "path": path,
                "slots": slots,
                "transcript": rec["transcript"]
            }
        )

    return slp


if __name__ == "__main__":
    print(get_slurp_sf(skip_exist=False, write_if_ne=True)[:10])
