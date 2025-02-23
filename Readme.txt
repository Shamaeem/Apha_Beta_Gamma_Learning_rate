


#Alpha
python ./src/main.py --optim-method SGD_tan1_Decay --alpha 0.999744 --nesterov --momentum 0.9 --weight-decay 0.0005  --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/tan1  --dataroot ./data

#Beta
python ./src/main.py --optim-method SGD_tan2_Decay --alpha 0.999744 --nesterov --momentum 0.9 --weight-decay 0.0005  --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/tan2  --dataroot ./data

#Gamma
python ./src/main.py --optim-method SGD_tan3_Decay --alpha 0.999744 --nesterov --momentum 0.9 --weight-decay 0.0005  --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/tan3  --dataroot ./data