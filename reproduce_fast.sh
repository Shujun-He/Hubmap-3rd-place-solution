cd train_code_fast/resnext50
for i in {0..4};do
  ../../venv/bin/python train.py --fold $i --nfolds 5 --batch_size 64 --epochs 50 --lr 1e-4 --expansion 0 --workers 16
done
cd ../../
