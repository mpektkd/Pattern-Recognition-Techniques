The present task deals with the music genre recognition and emotional dimensions extraction from spectograms / chromograms of musical pieces. 

- **Research Topic 1** (Simple Classification Models):<br />Firstly, we train some LSTMs, 2D CNNs in order to classify the music genres. We use Earcly Stopping and L2-Reguralization. 

- **Research Topic 2** (Regression Task):<br />Then, we use LSTM, CNN as Regression models to predict the emotional dimensions such as 'Valence', 'Energy', 'Danceability'. These models constitute the Baseline models for the comparison with the following models.

- **Research Topic 3** (Transfer Learning from classifcation to Regression):<br />Also, after that we use the best models of Topic 1 for Transfer Learning for regression task. We use (a) one-task model per emotion. The research interest here is that we use Spearman metric as Loss for training.

- **Research Topic 4** (Multitask Model):<br />Again for the best models, we use (b) multi-task models for regression. The research interest here is that we experiment with a weighted multiloss of 3 Losses.

In the case of trainable weights, weight has the form w = exp(-param), param(init) = [0,0,0] and
fin_loss = exp(-param)*loss+param ([here]([https://github.com/mpektkd/Pattern-Recognition-Techniques/blob/0416a725658d79622ea8f780b904e38e9f4dd641/Music%20Emotional%20Multitasking%20Regression%20Model%20using%20Spearman%20Metric/lib.py#L101](https://github.com/mpektkd/Pattern-Recognition-Techniques/blob/386d1454cf43408577682d1db175cfd8b6876bbc/Music%20Emotional%20Multitasking%20Regression%20Model%20using%20Spearman%20Metric/lib.py#L101))). Since the weight is a decreasing function this means that as
loss is positive the param tends to increase in order to converge the weight to zero. Because
however, our loss consists of the sum of spearman and mse, then this is done according to the order of
of magnitudes and negative. Then the param tends to decrease to increase the weight and drop the loss.
That is why negative values ​​arise in the weights. However the metrics are improved compared to Topic 2.
