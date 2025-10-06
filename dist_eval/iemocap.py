import os

label_dir = "/home/jpong/Workspace/jaeeewon/IEMOCAP/Session5"
# /dialog/EmoEvaluation
# /sentences/wav

"""
anns: {'Neutral': 1259, 'Happy': 1131, 'Sad': 800, 'Angry': 900}
1to4: {'neu': 1324, 'exc': 742, 'xxx': 1987, 'hap': 452, 'sur': 89, 'fea': 30, 'dis': 2, 'sad': 839, 'fru': 1468, 'ang': 933, 'oth': 3}
Neutral -> neu
Happy -> exc, hap
Sad -> sad
Angry -> ang
"""


def get_iemocap_er():
    ls = []
    # emotions = {}
    for label in os.listdir(os.path.join(label_dir, "dialog/EmoEvaluation")):
        if not label.endswith(".txt"):
            continue
        label_path = os.path.join(label_dir, "dialog/EmoEvaluation", label)
        with open(label_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("[") and "]" in line:
                splt = line.split("\t")
                t = splt[0].strip()  # [start - end]
                turn_name = splt[1].strip()  # Ses05F_impro01_F000
                emotion = splt[2].strip()  # neu
                # time_range = t[1 : t.index("]")].split(" - ")
                # start_time = float(time_range[0])
                # end_time = float(time_range[1])
                wav_path = os.path.join(
                    label_dir,
                    "sentences/wav",
                    label.replace(".txt", ""),
                    turn_name + ".wav",
                )
                # if emotion not in emotions:
                #     emotions[emotion] = 0
                # emotions[emotion] += 1
                ls.append({"path": wav_path, "emotion": emotion})
                # print(f"[{wav_path}] [{start_time:.4f}] [{end_time:.4f}] [{emotion}]")
    # print(emotions)
    return ls


print(get_iemocap_er()[:10])
