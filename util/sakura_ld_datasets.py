import os, librosa, numpy as np, soundfile as sf, argparse

from evaluation.data.sakura import SakuraDataProvider
from evaluation.types.redis_config import RedisConfig
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
    # python -m util.sakura_ld_datasets --model_name salmonn-7b --exp_id SLMN7-SKR-LD-NZ-EARLY-30s-B8 --type noised --pos early --target_len 30 --is_exp (--force)
    parser = argparse.ArgumentParser(description="sakura_ld dataset tool")
    parser.add_argument("--model_name", type=str, default="salmonn-7b", help="model name")
    parser.add_argument("--exp_id", type=str, required=True, help="experiment id")
    parser.add_argument("--type", type=str, required=True, help="type of ld dataset")
    parser.add_argument("--pos", type=str, required=True, help="position of ld dataset")
    parser.add_argument("--target_len", type=int, required=True, help="target length of ld dataset")
    parser.add_argument("--is_exp", action="store_true", help="whether to use partial dataset for experiment")
    parser.add_argument("--force", action="store_true", help="whether to force insert even if dataset exists")
    args = parser.parse_args()

    model_name = args.model_name
    exp_id = args.exp_id
    ld_type = args.type
    ld_pos = args.pos
    target_len = args.target_len
    is_exp = args.is_exp
    force = args.force

    data_provider = SakuraDataProvider(
        redis_cfg=RedisConfig(host="salmonn.hufs.jae.one", port=6379, db=11),
        key_prefix=f"{model_name}:{exp_id}",
        required_fields=["wav", "query"],
        filter={},
        skip_empty_warning=True,
    )

    assert force or not len(
        data_provider
    ), f"dataset already exists for the given model_name {model_name} and exp_id {exp_id}. use --force to override."

    ds = [
        {**d, "key": f"{data_provider.data_id}:{i}", "status": "initialized"}
        for i, d in enumerate(get_sakura_ld_ds(ld_type, ld_pos, target_len=target_len, is_exp=is_exp, gen=True))
    ]
    data_provider.insert_samples(ds)
    print(f"inserted {len(ds)} samples to redis for the given model_name {model_name} and exp_id {exp_id}.")
    print(args)

    # for target_len in [30]:
    #     for type in ld_conf["ld_keys"].keys():
    #         for pos in ld_conf["positions"]:
    #             ds = get_sakura_ld_ds(type, pos, target_len=target_len, is_exp=True, gen=True)
    #             print(ds[0])

    #             json_path = f"ann/sakura_ld/sakura_ld_{ld_conf['ld_keys'][type]}_{pos}_{target_len}.json"
    #             with open(json_path, "w") as f:
    #                 json.dump({"annotation": ds}, f, indent=4, ensure_ascii=False)
