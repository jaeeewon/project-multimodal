# 1. download common voice (v4)
```bash
mkdir tr
wget -O 'tr/cv-corpus-4-en.tar.gz' 'https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-4-2019-12-10/en.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250923%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250923T073344Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=1805d2c3f79fcc1b5e04515a64b17feeac4b0cc4cf9dc5b9f2bf9d9c15e6976fc45a7c59503c8a5ea339331085927b741f18cb58acee247fe06eaac5f5e3f5f6f71cb31f192a3bc8ca5982110f929414055822339148b13b578086593d9f7c2c3de4ea237de90a2abe2cb65382a145b9b5f44328ff2830e8a08de18b2cefea241d1797d0d9ab179989816974cd7325cba421e73ebd715e1aacaf61f99550e30e4bb257492781375de953a286295b4bcbb206c6f5174158e265d81308674b0a8a44f06ac553f99fc414a313f35db6e27e723be30449aded9925e5aed15b121f37a92a4c5aafb84939ac453cadec5e027749e81aa93383c7b687618409658159d7'
tar -xzvf "tr/cv-corpus-4-en.tar.gz" -C "tr"
rm -rf "tr/cv-corpus-4-en.tar.gz"
```

# 2. download translated data
```bash
wget -O "tr/covost_v2.en_de.tsv.tar.gz" https://dl.fbaipublicfiles.com/covost/covost_v2.en_de.tsv.tar.gz
wget -O "tr/covost_v2.en_ja.tsv.tar.gz" https://dl.fbaipublicfiles.com/covost/covost_v2.en_ja.tsv.tar.gz
wget -O "tr/covost_v2.en_zh-CN.tsv.tar.gz" https://dl.fbaipublicfiles.com/covost/covost_v2.en_zh-CN.tsv.tar.gz

tar -xzvf "tr/covost_v2.en_de.tsv.tar.gz" -C "tr"
tar -xzvf "tr/covost_v2.en_ja.tsv.tar.gz" -C "tr"
tar -xzvf "tr/covost_v2.en_zh-CN.tsv.tar.gz" -C "tr"

rm -rf "tr/covost_v2.en_de.tsv.tar.gz"
rm -rf "tr/covost_v2.en_ja.tsv.tar.gz"
rm -rf "tr/covost_v2.en_zh-CN.tsv.tar.gz"
```

# 3. run python code
> eval using `tr/covost_v2.en_<to>.tsv`