# demo program of LSTM using tensorflow  
This is the demo program of LSTM using tensorflow.  
It needs input sin(x) + e(noise),  and predict sin(x + dx).  


# dependency  
I confirmed operation only with..  
1)python==3.6.0    
2)tensorflow==1.8.0   
3)matplotlib==2.0.0  

# computation graph  
<img width="428" alt="tensorboard_graph" src="https://user-images.githubusercontent.com/15444879/41511026-3474c224-72aa-11e8-9a2e-ed65eae069e5.png">

# loss of test data  
<img width="357" alt="tensorboard_loss" src="https://user-images.githubusercontent.com/15444879/41511036-60846766-72aa-11e8-9b95-35259f4a9ba4.png">  

# R2 of test data  
<img width="365" alt="tensorboard_r2" src="https://user-images.githubusercontent.com/15444879/41511043-750803b4-72aa-11e8-8a38-6bfd3c797f15.png">  

# Prediction  
solid line is sinx, dashed line is predicted sin(x).  
ex1)  
![figure_mod02](https://user-images.githubusercontent.com/15444879/41525058-b6c03900-731a-11e8-9cd2-6e01e0a1e981.png)  

ex2)  
![figure_mod03](https://user-images.githubusercontent.com/15444879/41525076-c6f5dc80-731a-11e8-962d-cf9956931bc1.png)  
