python train.py -s ./data/white_background_70/ --config ./config/train.yaml --exp_id "one"
wait $!

python train.py -s ./data/white_background_70/ --config ./config/train_two.yaml --exp_id "two"
wait $!

python train.py -s ./data/white_background_100/ --config ./config/train_thr.yaml --exp_id "thr"
wait $!

python train.py -s ./data/white_background_100/ --config ./config/train_four.yaml --exp_id "four"
wait $!