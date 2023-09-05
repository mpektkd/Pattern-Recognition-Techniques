The purpose is the implementation of a voice processing and recognition system, with application in recognition
of individual words.

  - **Research Topic 1** (Dataset Construction): 
    </br>
    We apply Stratified split and normalization in the data points.

  - **Research Topic 2** (GMM-HMM):
    - A GMM-HMM model is constructed for each digit. First, the trainset is grouped
      depending on the digit they describe. Then, 4 states are selected for the HMM
      and 4 Gaussians for the GMM. We try General Mixture Model and Multivariate Models.
      The training is done using EM and for the hyperparameter values ​​chosen and for
      4 HMM states and 4 Gaussian distributions, the classifier has an accuracy rate of: 99.45 %
      on validation set.
    - Next, we optimize this model, by choosing the optimal hyperparameters
      using the optuna library. The number of HMM states, number of Gaussian distributions and number
      of repetitions during training are chosen as study variables. Its accuracy score is chosen
      as an optimization criterion and the hyperparameters that the optimal model has are stored in .pickle file.

            Hyperparameter                    Value
            HMM States                          3
            Number of Gaussians                 5
            Maximum number of iterations       300

      </br>
      The tuning of the optimal parameters is done by training our respective model in the trainset and evaluating it in the validation-set, to see its behavior in different
      scatter data. Finally, the model for these hyperparameters was trained on the trainset and tested
      over the test-set, achieving accuracy: 99%. From the following we find that in both cases the classifier performs very well
      results with minimal misclassifications as can be seen from the confusion matrices.
      
      </br>
      </br>
      
        <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/7c233456-5a86-417c-ad92-b7a5b48b663c" width="420" height="300">
        <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/bf9cdc66-b913-441f-a58e-23cb245ffc9e" width="420" height="300">

  
