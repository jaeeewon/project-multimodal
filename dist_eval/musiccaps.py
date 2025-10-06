import os
import pandas as pd

mc_dir = "/home/jpong/Workspace/jaeeewon/musiccaps"

# musiccaps-public.csv
# music_data


def get_musiccaps_mc(skip_missing=True):
    mc = pd.read_csv(os.path.join(mc_dir, "musiccaps-public.csv"))
    mc = mc[mc["is_audioset_eval"] == True]
    mc["path"] = mc["ytid"].apply(
        lambda ytid: os.path.join(mc_dir, "music_data", ytid + ".wav")
    )
    if skip_missing:
        mc = mc[mc["path"].apply(os.path.exists)]  # keep exists only
    else:
        missing = 0
        for _, m in mc.iterrows():
            if not os.path.exists(m["path"]):
                print("missing:", m["path"])
                missing += 1
        print(f"total missing: {missing} ({(missing / len(mc) * 100):.2f}%)") # total missing: 30 (1.05%)

    mc = mc[["caption", "path"]].to_dict("records")

    return mc


# print(get_musiccaps_mc()[:10])
