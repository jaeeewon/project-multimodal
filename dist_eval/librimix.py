import pandas as pd

"""
strategies for librimix inference & evaluation

test_clean_meta
/home/jpong/Workspace/jaeeewon/LibriMix_repo/metadata/Libri2Mix/libri2mix_test-clean.csv
mixture_ID,source_1_path,source_1_gain,source_2_path,source_2_gain,noise_path,noise_gain
7127-75947-0007_5683-32866-0022,test-clean/7127/75947/7127-75947-0007.flac,0.4180176287160039,test-clean/5683/32866/5683-32866-0022.flac,1.0441601282235726,tt/441c020k_0.31137_447c020y_-0.31137.wav,2.3888108943094184

test_mix_clean_meta
/home/jpong/Workspace/jaeeewon/LibriMix/Libri2Mix/wav16k/max/metadata/mixture_test_mix_clean.csv
mixture_ID,mixture_path,source_1_path,source_2_path,length
7127-75947-0007_5683-32866-0022,/home/jpong/Workspace/jaeeewon/LibriMix/Libri2Mix/wav16k/max/test/mix_clean/7127-75947-0007_5683-32866-0022.wav,/home/jpong/Workspace/jaeeewon/LibriMix/Libri2Mix/wav16k/max/test/s1/7127-75947-0007_5683-32866-0022.wav,/home/jpong/Workspace/jaeeewon/LibriMix/Libri2Mix/wav16k/max/test/s2/7127-75947-0007_5683-32866-0022.wav,87360

mixture_id = source1_source2

idea 1
test_mix_clean_meta의 mixture_ID, mixture_path를 {id, path}로 디비에 저장 및 추론
평가시 id에서 source1, source2 추출해 각각의 정답을 확인

idea 2 (선택)
test_mix_clean_meta의 mixture_ID, mixture_path와 추출된 각각의 정답을 {id, path, src1, src2}로 디비에 저장 및 추론
평가시 src1, src2로 각각의 정답을 확인
"""

test_mix_clean_meta_path = "/home/jpong/Workspace/jaeeewon/LibriMix/Libri2Mix/wav16k/max/metadata/mixture_test_mix_clean.csv"
librispeech_dir = "/home/jpong/Workspace/jaeeewon/LibriSpeech"

test_mix_clean_meta = pd.read_csv(test_mix_clean_meta_path)


def mid_to_ans(mid: str):
    # 7127-75947-0007_5683-32866-0022 -> (7127-75947-0007, 5683-32866-0022)
    s1, s2 = mid.split("_")
    spl1 = s1.split("-")
    spl2 = s2.split("-")

    d1 = spl1[:2]
    d2 = spl2[:2]

    # (7127-75947-0007, 5683-32866-0022) -> (test-clean/7127/75947/7127-75947-0007.flac, test-clean/5683/32866/5683-32866-0022.flac)
    # s1_path = f"test-clean/{'/'.join(d1)}/{s1}.flac"
    # s2_path = f"test-clean/{'/'.join(d2)}/{s2}.flac"

    # (7127-75947-0007, 5683-32866-0022) -> (test-clean/7127/75947/7127-75947.trans.txt, test-clean/5683/32866/5683-32866.trans.txt)
    d1_path = f"test-clean/{'/'.join(d1)}/{'-'.join(d1)}.trans.txt"
    d2_path = f"test-clean/{'/'.join(d2)}/{'-'.join(d2)}.trans.txt"

    # s1_path = f"{librispeech_dir}/{s1_path}"
    # s2_path = f"{librispeech_dir}/{s2_path}"
    d1_path = f"{librispeech_dir}/{d1_path}"
    d2_path = f"{librispeech_dir}/{d2_path}"

    with open(d1_path, "r") as f1, open(d2_path, "r") as f2:
        for _ in range(int(spl1[2])):
            next(f1)
        for _ in range(int(spl2[2])):
            next(f2)
        a1 = f1.readline().split(" ", 1)[1].strip()
        a2 = f2.readline().split(" ", 1)[1].strip()

    # print('s1_path:', s1_path)
    # print('s1_trans:', a1)
    # print('s2_path:', s2_path)
    # print('s2_trans:', a2)

    return a1, a2


def get_librimix_osr():
    lms = []

    # t1 = time.perf_counter()
    for idx, row in test_mix_clean_meta.iterrows():
        mid = row["mixture_ID"]
        a1, a2 = mid_to_ans(mid)
        lms.append({"id": mid, "path": row["mixture_path"], "ans1": a1, "ans2": a2})
    # print(f"elapsed: {time.perf_counter() - t1:.2f}s")

    return lms


if __name__ == "__main__":
    lms = get_librimix_osr()
    print(lms[:3])
