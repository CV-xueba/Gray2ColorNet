#! /bin/sh

if [ -f "./test.out" ]
then
  rm "./test.out"
fi
nohup python3 -u test.py --dataroot ./datasets/imagenet \
                         --use_D \
                         --preprocess none \
			                   --gpu_id 0 >test.out 2>&1 &

