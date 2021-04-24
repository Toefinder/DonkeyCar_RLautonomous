#!/bin/sh
nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/baselinecnn_nolanesegment_run1.h5" --debug_mode=0 --test --env_name="donkey-warren-track-v0" --max_ep_len=1000 --port 9091 --lane_segment=0 --keep_ratio=0 --image_rescale=0 > otherenv/baselinecnn_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/baselinecnn_lanesegment_run1.h5" --debug_mode=0 --test --env_name="donkey-warren-track-v0" --max_ep_len=1000 --port 9091 --lane_segment=1 --keep_ratio=0 --image_rescale=0 > otherenv/baselinecnn_lanesegment_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit1_nolanesegment_run1.h5" --debug_mode=0 --test --max_ep_len=1000 --port 9091 --lane_segment=0 --keep_ratio=0 --image_rescale=0 --model_name="cnnedit1" --env_name="donkey-warren-track-v0" > otherenv/cnnedit1_nolanesegment_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit1_lanesegment_run1.h5" --debug_mode=0 --test --max_ep_len=1000 --port 9091 --lane_segment=1 --keep_ratio=0 --image_rescale=0 --model_name="cnnedit1" --env_name="donkey-warren-track-v0" > otherenv/cnnedit1_lanesegment_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit1_lstm_nolanesegment_run1.h5" --debug_mode=0 --test --max_ep_len=1000 --port 9091 --lane_segment=0 --keep_ratio=0 --image_rescale=0 --model_name="cnnedit1_lstm" --env_name="donkey-warren-track-v0" > otherenv/cnnedit1_lstm_nolanesegment_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit1_lstm_lanesegment_run1.h5" --debug_mode=0 --test --max_ep_len=1000 --port 9091 --lane_segment=1 --keep_ratio=0 --image_rescale=0 --model_name="cnnedit1_lstm" --env_name="donkey-warren-track-v0" > otherenv/cnnedit1_lstm_lanesegment_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit1_lanesegment_1frame_run1.h5" --debug_mode=0 --test --max_ep_len=1000 --port 9091 --lane_segment=1 --keep_ratio=0 --image_rescale=0 --img_channel=1 --model_name="cnnedit1" --env_name="donkey-warren-track-v0" > otherenv/cnnedit1_lanesegment_1frame_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit2_lanesegment_4frame_run1.h5" --debug_mode=0 --test --max_ep_len=1000 --port 9091 --lane_segment=1 --keep_ratio=0 --image_rescale=0 --img_channel=4 --model_name="cnnedit2" --env_name="donkey-warren-track-v0" > otherenv/cnnedit2_lanesegment_4frame_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/transferlearning_mobilenetv2_nolanesegment_4frame_run1.h5" --debug_mode=0 --max_ep_len=1000 --port 9091 --lane_segment=0 --keep_ratio=0 --image_rescale=0  --model_name="transfer_mobilenetv2" --test --env_name="donkey-warren-track-v0" --img_channels=4 --img_size=96 --color_channels=3 > otherenv/transferlearning_mobilenetv2_nolanesegment_4frame_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/transferlearning_mobilenetv2_lanesegment_4frame_run1.h5" --debug_mode=0 --test --env_name="donkey-warren-track-v0" --max_ep_len=1000 --port 9091 --lane_segment=1 --keep_ratio=0  --model_name="transfer_mobilenetv2"  --img_channels=4 --img_size=96 --color_channels=1 > otherenv/transfer_mobilenetv2_lanesegment_4frame_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit1_lstm_reoder_lanesegment_run1.h5" --debug_mode=0 --test --env_name="donkey-warren-track-v0" --max_ep_len=1000 --port 9091 --lane_segment=1 --keep_ratio=0 --image_rescale=0 --model_name="cnnedit1_lstm_reorder" > otherenv/cnnedit1_lstm_reorder_lanesegment_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit1_lstm_reoder_nolanesegment_run1.h5" --debug_mode=0 --test --env_name="donkey-warren-track-v0" --max_ep_len=1000 --port 9091 --lane_segment=0 --keep_ratio=0 --image_rescale=0 --model_name="cnnedit1_lstm_reorder" > otherenv/cnnedit1_lstm_reorder_nolanesegment_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/transferlearning_mobilenetv2_lstm_nolanesegment_4frame_run1.h5" --debug_mode=0 --test --env_name="donkey-warren-track-v0" --max_ep_len=1000 --port 9091 --lane_segment=0 --keep_ratio=0  --model_name="transfer_mobilenetv2_lstm"  --img_channels=4 --img_size=96 --color_channels=3 > otherenv/transfer_mobilenetv2_lstm_nolanesegment_4frame_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/transferlearning_mobilenetv2_lstm_nolanesegment_4frame_run1.h5" --debug_mode=0 --test --env_name="donkey-warren-track-v0" --max_ep_len=1000 --port 9091 --lane_segment=0 --keep_ratio=0  --model_name="transfer_mobilenetv2_lstm"  --img_channels=4 --img_size=96 --color_channels=3 > otherenv/transfer_mobilenetv2_lstm_lanesegment_4frame_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit3_lanesegment_run1.h5" --debug_mode=0 --test --env_name="donkey-warren-track-v0" --max_ep_len=1000 --port 9091 --lane_segment=1 --keep_ratio=0 --image_rescale=0 --model_name="cnnedit3" > otherenv/cnnedit3_lanesegment_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit4_lanesegment_run1.h5" --debug_mode=0 --test --env_name="donkey-warren-track-v0" --max_ep_len=1000 --port 9091 --lane_segment=1 --keep_ratio=0 --image_rescale=0 --model_name="cnnedit4" > otherenv/cnnedit4_lanesegment_run1.out ;

nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 5 --model "model/cnnedit4_lstm_lanesegment_run1.h5" --debug_mode=0 --test --env_name="donkey-warren-track-v0" --max_ep_len=1000 --port 9091 --lane_segment=1 --keep_ratio=0 --image_rescale=0 --model_name="cnnedit4_lstm" > otherenv/cnnedit4_lstm_lanesegment_run1.out ;










