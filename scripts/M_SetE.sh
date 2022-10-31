cd `dirname $0`; cd ..;


CUDA_VISIBLE_DEVICES=1 python M_SetE.py --run_id normal --mode train

