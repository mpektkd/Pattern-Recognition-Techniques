The present task deals with the music genre recognition and emotional dimensions extraction from spectograms / chromograms of musical pieces. There are 2 datasets for genre recognition, one with non-beat synced spectograms / chromograms an one synced. Through experiments we conclude that the synced one gives better results. 

- **Research Topic 1** (Simple Classification Models):<br />Firstly, we train some LSTMs, 2D CNNs in order to classify the music genres(10 classes). We use Earcly Stopping and L2-Reguralization. We show the results only of the fused beat-synced dataset(spectograms concatenated with chromograms), as it the best case.
   - LSTM:

                        precision    recall  f1-score   support
          
                     0       0.00      0.00      0.00        40
                     1       0.41      0.17      0.25        40
                     2       0.25      0.55      0.35        80
                     3       0.31      0.56      0.40        80
                     4       0.00      0.00      0.00        40
                     5       0.00      0.00      0.00        40
                     6       0.30      0.08      0.12        78
                     7       0.00      0.00      0.00        40
                     8       0.28      0.60      0.38       103
                     9       0.00      0.00      0.00        34
          
              accuracy                           0.29       575
             macro avg       0.16      0.20      0.15       575
          weighted avg       0.20      0.29      0.21       575

   - 2D CNN:


                       precision    recall  f1-score   support
          
                     0       0.08      0.03      0.04        40
                     1       0.37      0.47      0.42        40
                     2       0.34      0.54      0.42        80
                     3       0.32      0.57      0.41        80
                     4       0.40      0.20      0.27        40
                     5       0.00      0.00      0.00        40
                     6       0.53      0.38      0.44        78
                     7       0.19      0.07      0.11        40
                     8       0.36      0.46      0.40       103
                     9       0.17      0.09      0.12        34
          
              accuracy                           0.35       575
             macro avg       0.28      0.28      0.26       575
          weighted avg       0.31      0.35      0.31       575
     
- **Research Topic 2** (Regression Task):<br />Then, we use LSTM, CNN as Regression models to predict the emotional dimensions such as 'Valence', 'Energy', 'Danceability'. These models constitute the Baseline models for the comparison with the following models in topics 3, 4.The research interest here is that we use Spearman metric as Loss for training. We provide some results for the CNN model, that gives better results, for the emotion 'Valence' that shows the less correlation based on the Spearman Metric, in order to show the improvement in Topic 4.

  <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/7926dd05-cd98-4276-b4d7-689b5e7c97d4" width="500" height="300">
  <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/f07710f6-fa3a-4786-8c1f-fe6dfef0eaaa" width="400" height="300">
  
   - **Notes**:<br />
      ✓ By adding only mse-loss we get values ​​close to 0.5 which is not desirable, because in
        in this case the model finds the easiest solution since most inputs
        is from 0.3 to 0.8. To avoid this, Spearman-loss is added to have
        results across the range from 0 to 1 and get consistent results.<br />
      ✓ The specific implementation of Spearman loss is differentiable as its most basic implementation
        Spearman-loss does not accept gradients. That's why the "fast_soft_sort" library was chosen.<br />
- **Research Topic 3** (Transfer Learning from classifcation to Regression):<br />Also, after that we use the best model of Topic 1 for Transfer Learning for regression task e.g. CNN. We train a CNN one-task model per emotion. We show the results below:
  
  <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/c232749b-f82e-4beb-b5c1-17232e353aff" width="400" height="300">
  <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/efc4bef1-2d20-40fc-a95d-dceb3e4d875e" width="400" height="300">


- **Research Topic 4** (Multitask Model):<br />Again for the best models, we use (b) multi-task models for regression. The research interest here is that we experiment with a weighted multiloss of 3 Losses. In the case of trainable weights, weight has the form w = exp(-param), param(init) = [0,0,0] and
fin_loss = exp(-param)*loss+param ([here](https://github.com/mpektkd/Pattern-Recognition-Techniques/blob/d7d926e3a669cb2138a6dfd381e71b92448c49ea/Music%20Emotional%20Multitasking%20Regression%20Model%20using%20Spearman%20Metric/lib.py#L101)). Since the weight is a decreasing function this means that as loss is positive the param tends to increase in order to converge the weight to zero. Because however, our loss consists of the sum of spearman and mse, then this is done according to the order of of magnitudes and negative. Then the param tends to decrease to increase the weight and drop the loss. That is why negative values ​​arise in the weights. However the metrics are improved compared to Topic 2. Below we present the results for the multitasking CNN. We can notice the improvement for the emotion 'Valence'.

<img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/36ecfd5e-2479-42c8-a13f-df7fd34b678f" width="450" height="300">
<img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/feabfdab-8979-4157-9e81-50fbb107093a" width="1000" height="350">
