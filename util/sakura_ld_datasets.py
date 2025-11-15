import os, json, librosa, numpy as np, soundfile as sf
from .sakura_datasets import get_sakura_ds


ld_conf = {
    "ld_path": "/home/jpong/Workspace/jaeeewon/sakura_ld",
    "positions": ["early", "middle", "late"],
    "ld_keys": {
        "zero-padded": "zp",
        "noised": "nz",
        "source": "src",
    },
    "target_sr": 16000,  # sr=16k
}


def get_sakura_ld_ds(type: str, pos: str, target_len: int, is_exp=False, gen=False):
    positions = ld_conf["positions"]
    ld_keys = ld_conf["ld_keys"]
    target_sr = ld_conf["target_sr"]
    ld_path = ld_conf["ld_path"]
    types = ld_keys.keys()

    assert type in types, f"type should be one of {types}"
    assert pos in positions or type == "source", f"position should be one of {positions}"

    for t in types:
        dir_path = os.path.join(ld_path, pos, t)
        os.makedirs(dir_path, exist_ok=True)

    sakura = get_sakura_ds(is_exp=is_exp)
    sakura_ld = []

    fillers = {
        "zero-padded": lambda fill_len: np.zeros(fill_len),
        "noised": lambda fill_len: 0.005 * np.random.randn(fill_len),
    }

    for data in sakura:
        # <update_path>
        if type != "source":
            org = data["wav"]
            file_name = os.path.basename(org).replace(".wav", f"_{target_len}s.wav")
            data["wav"] = os.path.join(ld_path, pos, type, file_name)
            if gen and not os.path.exists(data["wav"]):
                y, _ = librosa.load(org, sr=target_sr)
                fill_len = target_sr * target_len - len(y)
                assert fill_len > 0, f"original length must be shorter than {target_len} sec"

                filler = fillers[type](fill_len)
                if pos == "early":
                    y_filled = np.concatenate([y, filler])
                elif pos == "middle":
                    y_filled = np.concatenate([filler[: fill_len // 2], y, filler[fill_len // 2 :]])
                elif pos == "late":
                    y_filled = np.concatenate([filler, y])

                sf.write(data["wav"], y_filled, target_sr)
        # </update_path>

        data["ld_pos"] = pos if type != "source" else "original"
        data["ld_type"] = type

        sakura_ld.append(data)

    return sakura_ld


if __name__ == "__main__":
    for target_len in [30]:
        for type in ld_conf["ld_keys"].keys():
            for pos in ld_conf["positions"]:
                ds = get_sakura_ld_ds(type, pos, target_len=target_len, is_exp=True, gen=True)
                print(ds[0])

                json_path = f"ann/sakura_ld/sakura_ld_{ld_conf['ld_keys'][type]}_{pos}_{target_len}.json"
                with open(json_path, "w") as f:
                    json.dump({"annotation": ds}, f, indent=4, ensure_ascii=False)
