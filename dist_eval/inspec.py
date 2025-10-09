import os, soundfile as sf
from tts import TTS
from datasets import load_dataset
from tqdm import tqdm

ip_dir = "/home/jpong/Workspace/jaeeewon/inspec/wav"


def get_inspec_ke(skip_exist=False):
    ds = load_dataset("taln-ls2n/inspec", split="test")
    ip = ds.to_pandas()
    ip["path"] = ip["id"].apply(
        lambda id: os.path.join(ip_dir, id + ".wav")
    )  # clova returns mp3 formatted sound
    ip["keyphrases"] = ip["keyphrases"].apply(lambda k: "; ".join(k))
    ip = ip[["id", "path", "abstract", "keyphrases"]].to_dict("records")
    if not skip_exist:
        for rec in ip:
            assert os.path.exists(rec["path"]), f"not exists: {rec}"

    return ip


def gen_tts():
    kes = get_inspec_ke(skip_exist=True)
    tts = TTS()
    for ke in tqdm(kes):
        tts.edge_generate(ke["abstract"], ke["path"])


if __name__ == "__main__":
    print(get_inspec_ke()[0])
    # gen_tts()
