import os

vc1_dir = "/home/jpong/Workspace/jaeeewon/VoxCeleb1/wav"
# wav


def get_voxceleb1_sv():
    ls = []
    for dir in os.listdir(vc1_dir):
        if not dir.startswith("id"):
            print("invalid dir:", dir)
            continue
        for subdir in os.listdir(os.path.join(vc1_dir, dir)):
            for file in os.listdir(os.path.join(vc1_dir, dir, subdir)):
                if not file.endswith(".wav"):
                    continue
                path = os.path.join(vc1_dir, dir, subdir, file)
                ls.append({"path": path})

    return ls


print(get_voxceleb1_sv()[:10])
