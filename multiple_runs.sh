#!/bin/bash
mkdir print_logs

MODEL=resnet

echo "RUN 0 - 8"
nohup ./run.sh 0 8 $MODEL 0 &
sleep 1

echo "RUN 8 - 16"
nohup ./run.sh 8 16 $MODEL 0 &
sleep 1

echo "RUN 16 - 24"
nohup ./run.sh 16 24 $MODEL 0 &
sleep 1

echo "RUN 24 - 32"
nohup ./run.sh 24 32 $MODEL 0 &
sleep 1

echo "RUN 32 - 40"
nohup ./run.sh 32 40 $MODEL 0 &
sleep 1

echo "RUN 40 - 48"
nohup ./run.sh 40 48 $MODEL 1 &
sleep 1

echo "RUN 48 - 56"
nohup ./run.sh 48 56 $MODEL 1 &
sleep 1

echo "RUN 56 - 64"
nohup ./run.sh 56 64 $MODEL 1 &
sleep 1

echo "RUN 64 - 72"
nohup ./run.sh 64 72 $MODEL 1 &
sleep 1

echo "RUN 72 - 81"
nohup ./run.sh 72 81 $MODEL 1 &
sleep 1
