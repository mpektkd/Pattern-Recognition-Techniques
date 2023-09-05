The purpose is of this study to extract suitable acoustic features
from voice data, using appropriate python packages, as well as analysis and visualization
so that they understand and derive useful information from them. The features in question
is essentially a number of cepstrum coefficients extracted after analyzing the signals with a
specially designed filter bank. This array is inspired by psychoacoustics
studies. The dataset we process concerns the recognition of individual digits (isolated
digits) in English. The data you will use contains utterances of 9 out of 15 digits
different speakers in separate .wav files. Totally, there are 133 files and 15 announcers.

- **Research Topic 1**(Data Extraction):</br>
  In this step, the MFCCs are extracted as well as their first and second derivatives, deltas
  and delt-deltas. For each audio file is selected window length 25 ms and step 10 ms and 13
  different features are extracted. Obviously because files have different time lengths, the
  resulting arrays will have dimensions of the format (W x 13), where W depends each time on the respective audio file.
  - By visualizing the histograms of the first 2 MFCCs of the random digits '6', '9' we
    observe that for all utterances almost all values are clustered around zero. In in some utterances a
    greater width of the histograms is observed while in some others longer. These characteristics depend
    directly on the announcer and the need his voice.
  - Through Correlation Matrices, we find that the characteristic MFSCs are quite more
    correlated with respect to MFCCs and there is an obvious connection between them. For this 
    purpose in practice MFCCs are preferred because the characteristics are generally preferred to be
    as unrelated as possible to get the maximum possible information.
  
- **Research Topic 2** (Feature Visualization and Reduction):
  - For each utterance we concatenate the MFCCs and deltas and delta-deltas and for each
    from these we calculate the mean and the standard deviation to form new vectors.
    With this process we end up with two tables, one for the averages
    terms and one for the standard deviations, of dimensions (133 x 39) each.
    We provide below the scatter plots of the 'mean' and 'standard deviation' of the MFCCs
    for the first two dimensions.

    <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/860c6d3e-3605-48a7-ad63-5c1a5bc6c45e" width="420" height="300">
    <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/a4979265-acc7-4b5d-8f6b-ed3822a1bf02" width="420" height="300">
    </br>
    From the above diagrams, we can see that in general the characteristics are not distinct
    separable into the classes to which they belong. However, in some cases such as for example for
    the classes of digits 5 and 6 for the case of the means vectors and for the classes of
    digits 4 and 5 for the case of the standard deviation vectors, we have an obvious separation
    in relation to the rest.In general, the characteristics of the same classes are close in space,
    but there are no clear areas of separation between them.

  - Also, we apply reduction using PCA choosing as the final goal the existence of only 2
    dimensions. So for the two categories of vectors we re-form the scatter plots and we have:
    <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/41060e39-e877-48c5-ba60-e86a0a01c44e" width="420" height="300">
    <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/c22a8f73-260d-4e7c-906c-66d81be29092" width="420" height="300">
    </br>
    While in terms of the percentage of original variance retained by the resulting components, we have:

                                                        Percentage

                Vector Category              1st  Vector     2nd Vector
    
                Mean Vector                   0.66979977      0.14992746
                Std Vector                    0.69178909      0.15685564
    </br>
    Observing the resulting diagrams for the case of the two-dimensional display,
    we find a slightly improvement for feature distinction. With regard to 
    the percentage of the dispersion of the initials components, for both 
    categories the first vector holds the largest part of that rate.
  - Looking at the 3D scatter plot, we see that the classes now occupy one
    different region in space, compared to the one seen in the two-dimensional case. The third
    dimension gives the freedom of spatial arrangement and not simply flat. Thus the classes are separated
    better.

    </br>
    **Notes**:
    - The variance ratio for each component indicates the percentage of dispersion of the originals
      of dimensions, which it retains after the dimensional reduction. Also, to mention here that, the
      variance ratio is proportional to the measure of the "eigenvalue" of each component, which
      increases as the direction of the corresponding component approaches that of the initial maximum
      dispersion. (As shown in the picture below)

    <p align="center">
      <img src="https://github.com/mpektkd/Pattern-Recognition-Techniques/assets/62422421/32a82b85-dee1-453b-aedf-751ac677f432"  width="600" height="300">
    </p>
    
- **Research Topic 3** (Baseline Models for Digit Classification):</br>
  - In this step, various classifiers are trained and evaluated for various dataset adgustments. The results can be found in the notebook.
    Examining the above results we can conclude that the dimensionality reduction and
    in both cases it does not seem to give better results, since for most
    classifiers the accuracy and f1-score metrics decrease. Also, comparing the results that
    we get from vertical and horizontal stacking, we seem to get better results at
    case of the latter. </br>
  - Finally, we add additional features to the original data set. For
    in this particular case, the addition of the zero-crossing-rate was chosen. Comparing the results
    of the first and last case, it seems that the adding extra features improves the performance of
    classifiers. Specifically, in our case adding just one extra attributes to the data table is essential
    improving the performance of classifiers.
  - One of the baselines in a custom implementation of Gaussian NB.
