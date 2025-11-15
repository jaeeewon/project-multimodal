#!/bin/bash

# the code under mainly written by ChatGPT-5.1

# 예시:
#   데이터 인서트:
#     bash sakura_ld.sh --insert
#   exp 실행 (한 번 실행, 내부에서 여러 exp_ids 처리):
#     bash sakura_ld.sh --exp "cuda:0 cuda:1"

MODE=$1          # --insert 또는 --exp
DEVICES_RAW=$2   # --exp 모드에서만 사용: "cuda:0 cuda:1" 같은 문자열

if [[ "$MODE" != "--insert" && "$MODE" != "--exp" ]]; then
  echo "Usage: $0 [--insert | --exp] [devices]"
  exit 1
fi

if [[ "$MODE" == "--exp" && -z "$DEVICES_RAW" ]]; then
  echo "--exp requires argument 'devices'"
  echo "e.g. bash sakura_ld.sh --exp \"cuda:0 cuda:1\""
  exit 1
fi

MODEL="salmonn-7b"
TARGET_LEN=30
BATCH=5

mkdir -p ./logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate salmonn

declare -A LD_KEYS
LD_KEYS["zero-padded"]="zp"
LD_KEYS["noised"]="nz"
LD_KEYS["source"]="src"

TYPES=("zero-padded" "noised" "source")
POSITIONS=("early" "middle" "late")

EXP_IDS=()
EXP_TYPES=()
EXP_POSITIONS=()

for TYPE in "${TYPES[@]}"; do
  for POS in "${POSITIONS[@]}"; do
    KEY=${LD_KEYS[$TYPE]}
    EXP_ID="SKR-LD-${KEY^^}-${POS^^}-30s-B${BATCH}"
    EXP_IDS+=("${EXP_ID}")
    EXP_TYPES+=("${TYPE}")
    EXP_POSITIONS+=("${POS}")
  done
done

TOTAL_RUNS=${#EXP_IDS[@]}  # 9

if [[ "$MODE" == "--insert" ]]; then
  RUN_IDX=0

  for ((i=0; i<${TOTAL_RUNS}; i++)); do
    EXP_ID="${EXP_IDS[$i]}"
    TYPE="${EXP_TYPES[$i]}"
    POS="${EXP_POSITIONS[$i]}"

    TS=$(date +"%y%m%d-%H%M%S")
    LOGFILE="./logs/${MODEL}:${EXP_ID}_${TS}.log"

    RUN_IDX=$((RUN_IDX + 1))

    echo "==============================================" | tee "$LOGFILE"
    echo "MODE       = --insert" | tee -a "$LOGFILE"
    echo "RUN        = ${RUN_IDX}/${TOTAL_RUNS}" | tee -a "$LOGFILE"
    echo "TYPE       = ${TYPE}" | tee -a "$LOGFILE"
    echo "POSITION   = ${POS}" | tee -a "$LOGFILE"
    echo "EXP_ID     = ${EXP_ID}" | tee -a "$LOGFILE"
    echo "LOGFILE    = ${LOGFILE}" | tee -a "$LOGFILE"
    echo "START_TIME = $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOGFILE"
    echo "==============================================" | tee -a "$LOGFILE"

    (
      python -m util.sakura_ld_datasets \
        --model_name "${MODEL}" \
        --exp_id "${EXP_ID}" \
        --type "${TYPE}" \
        --pos "${POS}" \
        --target_len "${TARGET_LEN}" \
        --force
        # --is_exp \
    ) >> "$LOGFILE" 2>&1

    STATUS=$?

    if [[ $STATUS -ne 0 ]]; then
      echo "" | tee -a "$LOGFILE"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOGFILE"
      echo "ERROR: Python execution failed for EXP_ID=${EXP_ID}" | tee -a "$LOGFILE"
      echo "STATUS CODE = $STATUS" | tee -a "$LOGFILE"
      echo "Stopping pipeline." | tee -a "$LOGFILE"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOGFILE"
      exit 1
    fi

    echo "" | tee -a "$LOGFILE"
    echo "finished successfully" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"

  done

  exit 0
fi

if [[ "$MODE" == "--exp" ]]; then
  TS=$(date +"%y%m%d-%H%M%S")
  DEV_TAG="${DEVICES_RAW// /_}"   # 공백을 '_'로 치환해 파일명에 사용
  LOGFILE="./logs/${MODEL}:SKR-LD-ALL_${DEV_TAG}_${TS}.log"

  echo "==============================================" | tee "$LOGFILE"
  echo "MODE       = --exp" | tee -a "$LOGFILE"
  echo "DEVICES    = ${DEVICES_RAW}" | tee -a "$LOGFILE"
  echo "MODEL      = ${MODEL}" | tee -a "$LOGFILE"
  echo "BATCH      = ${BATCH}" | tee -a "$LOGFILE"
  echo "N_EXP_IDS  = ${TOTAL_RUNS}" | tee -a "$LOGFILE"
  echo "LOGFILE    = ${LOGFILE}" | tee -a "$LOGFILE"
  echo "START_TIME = $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOGFILE"
  echo "==============================================" | tee -a "$LOGFILE"

  # --exp_ids a b c ... 형태로 모두 전달
  python -m evaluation.sakura_exp \
    --devices ${DEVICES_RAW} \
    --exp_ids "${EXP_IDS[@]}" \
    --model_name "${MODEL}" \
    --batch_size "${BATCH}" \
    --skip_confirm >> "$LOGFILE" 2>&1

  STATUS=$?

  if [[ $STATUS -ne 0 ]]; then
    echo "" | tee -a "$LOGFILE"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOGFILE"
    echo "ERROR: sakura_exp failed." | tee -a "$LOGFILE"
    echo "STATUS CODE = $STATUS" | tee -a "$LOGFILE"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOGFILE"
    exit 1
  fi

  echo "" | tee -a "$LOGFILE"
  echo "All experiments finished successfully." | tee -a "$LOGFILE"
  exit 0
fi
