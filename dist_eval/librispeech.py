import os, json

libri_path = "/home/jpong/Workspace/jaeeewon/LibriSpeech"


def get_librispeech_list():
    ds = {"test-clean": [], "test-other": []}

    ignore_ls = [".DS_Store", "LICENSE.TXT", ".complete"]
    ignore = lambda x: x in ignore_ls

    for set in ds:
        root_dir = f"{libri_path}/{set}"
        for i in os.listdir(root_dir):
            if not ignore(i):
                path2 = f"{root_dir}/{i}"
                for ii in os.listdir(path2):
                    if not ignore(ii):
                        path = f"{root_dir}/{i}/{ii}"
                        with open(f"{path}/{i}-{ii}.trans.txt", encoding="utf-8") as f:
                            for line in f:
                                splt = line.strip().split(" ", maxsplit=1)
                                filename = f"{path}/{splt[0]}.flac"
                                answer = splt[1]
                                ds[set].append({"path": filename, "sentence": answer})
    return ds


def get_librispeech_pr():
    # ensure you have repr_exp/table3/LibriSpeech/LS_testclean_pr.json
    with open("repr_exp/table3/LibriSpeech/LS_testclean_pr.json", "r") as anns_f:
        anns = json.load(anns_f)["annotation"]

    return anns


if __name__ == "__main__":
    get_librispeech_list()
