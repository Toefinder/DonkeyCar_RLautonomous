# Implementation of paper titled "Performance of Deep Reinforcement Learning Models in End-to-End Autonomous Driving"
Experiments are conducted on Donkey Car Simulator. The code is mainly adapted from gym-donkeycar example code.  

Mentor: Mr Mo Xiaoyu\
References: https://github.com/tawnkramer/gym-donkeycar/tree/master/examples/reinforcement_learning 

## Structure of the folder:
/projects/gym-donkeycar/examples/reinforcement_learning: folder containing all the code edited by me  

Important files in this folder:  
&nbsp;&nbsp;&nbsp;ddqn.py: the main file used to run all models, but note that it fixes the throttle and only allows steering to be controlled   
&nbsp;&nbsp;&nbsp;models.py: build models that is used by ddqn.py   
&nbsp;&nbsp;&nbsp;ddqn_throttle.py: attempt to use the same algorithm to control both steering and throttle   
&nbsp;&nbsp;&nbsp;imageprocess.py: contains code for edge detection   
  
&nbsp;&nbsp;&nbsp;ppo_train.py: not written and not used by me 

## Requirements  
It is best to run the experiments in background with nohup on a Linux machine.  
1) Install the simulator from https://docs.donkeycar.com/guide/simulator/  
2) Install the packages from environment.yml 

### Running the experiments 
First, cd to /projects

To run Exact Non-federated Shapley for 5 users, for case 1 (equal distribution with same size), and 10 global epochs:     
```nohup python gym-donkeycar/examples/reinforcement_learning/ddqn.py --gpu 0 --eps 500 --model "model/transferlearning_mobilenetv2_lstm_nolanesegment_4frame_run1.h5" --debug_mode=0 --max_ep_len=1000 --port 9080 --lane_segment=0 --keep_ratio=0  --model_name="transfer_mobilenetv2_lstm"  --img_channels=4 --img_size=96 --color_channels=3 > transfer_mobilenetv2_lstm_nolanesegment_4frame_run1.out & ```

## Options 
"--model", type=str, default="rl_driver.h5", help="name of the model file"   
"--test", action="store_true", help="agent uses learned model to navigate env"  
"--port", type=int, default=9091, help="port to use for websockets"  
"--throttle", type=float, default=0.3, help="constant throttle for driving"  
"--env_name", type=str, default="donkey-generated-track-v0", help="name of donkey sim environment", choices=env_list  
"--gpu", type=int, default=0, help="name of GPU to use"  
"--debug_mode", type=int, default=0, help="debug mode"  
"--eps", type=int, default=500, help="number of episodes to train for"  
"--max_ep_len", type=int, default=1000, help="maximum length per episode"  
"--lane_segment", type=int, default=0, help="whether to perform lane segmentation"  
"--keep_ratio", type=int, default=0, help="whether to keep the image aspect ratio by padding before resizing"  
"--image_rescale", type=int, default=0, help="whether to rescale the image pixels before feeding it into the CNN"  
"--model_name", type=str, default="baseline", help="name of the model architecture to use"  
"--img_channels", type=int, default=4, help="number of image frames to stack"  
"--color_channels", type=int, default=1, help="number of colors in each image, can be either 1 or 3"  
"--img_size", type=int, default=80, help="size of square image"     
