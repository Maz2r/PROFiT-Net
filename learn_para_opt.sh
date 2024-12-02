#!/bin/bash

# 배치 사이즈와 학습률 배열 정의
batch_sizes=(128 256 512)
learning_rates=(0.0001 0.0005 0.001)

# 각 조합당 반복 횟수
repeat=3

# 총 실행 횟수 계산
total_runs=$(( ${#batch_sizes[@]} * ${#learning_rates[@]} * repeat ))
current_run=1

# 배치 사이즈 반복문
for batch_size in "${batch_sizes[@]}"; do
    # 학습률 반복문
    for learning_rate in "${learning_rates[@]}"; do
        # 반복 횟수만큼 실행
        for ((i=1; i<=repeat; i++)); do
            echo "총 $total_runs 번 중 $current_run 번째 실행: batch_size=$batch_size, learning_rate=$learning_rate, 반복=$i"

            # 이전 프로세스가 끝날 때까지 대기
            while true; do
                # train_2d_update.py 프로세스가 실행 중인지 확인
                if ps -ef | grep train_2d_update.py | grep -v grep > /dev/null; then
                    # 실행 중이면 30초 대기
                    sleep 30
                else
                    # 실행 중이 아니면 반복문 탈출
                    break
                fi
            done

            # 학습 프로세스 시작
            python src/train/pytorch/train_2d_update.py exp_fe --batch_size $batch_size --learning_rate $learning_rate --epochs 500 --target_mae 0.11 --target_mae_deviation 0.005 --patience 3

            # 현재 실행 번호 증가
            current_run=$((current_run + 1))
        done
    done
done

echo "모든 실행이 완료되었습니다."
