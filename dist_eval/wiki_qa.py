import os, soundfile as sf
from tts import TTS
from datasets import load_dataset
from tqdm import tqdm

wq_dir = "/home/jpong/Workspace/jaeeewon/wiki_qa/wav"


def get_wikiqa_sqqa(skip_exist=False):
    ds = load_dataset("microsoft/wiki_qa", split="test")
    wq = ds.to_pandas()
    wq = wq[["question_id", "question"]]
    wq["path"] = wq["question_id"].apply(lambda id: os.path.join(wq_dir, id + ".wav"))
    wq = wq.drop_duplicates().to_dict("records")

    if not skip_exist:
        for rec in wq:
            assert os.path.exists(rec["path"]), f"not exists: {rec}"
    return wq


def gen_tts():
    wqs = get_wikiqa_sqqa(skip_exist=True)
    tts = TTS()
    for wq in tqdm(wqs):
        tts.edge_generate(wq["question"], wq["path"])


if __name__ == "__main__":
    # print(get_wikiqa_sqqa()[:10])
    gen_tts()
