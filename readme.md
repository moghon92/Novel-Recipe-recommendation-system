1. **Motivation**

As a result of coronavirus pandemic, more people have been cooking at home than the past. Some surveys conducted among consumers [1] [2] have found that the intention to keep up with home cooking is especially strong among younger demographics. Through this project I would like to encourage the younger demographics to continue to experiment in the kitchen by recommending “novel recipes” that are of interest to them and that they have likely not tried before. This will hopefully help keep the younger demographics engaged and excited to explore new cooking styles.

1. **Problem Statement**

Today, there are many applications which enable home cooks to explore different recipes and follow laid out instructions to cook desired recipes. So far, applications mostly take only ingredients as an input and recommend a few recipes based on the user’s past activity [3][4]. However, this has the problem of recommending recipes that are very similar to the ones the user has already tried or favorited, and leaves very little room for recommending novel recipes that will help keep the user engaged and excited to try new things. In this project, I will build a “Novel recipe recommendation” application which will balance between recommending novel recipes that are not very familiar to the user, but at the same time have high likelihood of being rated highly by the user.

1. **Data Source:**

The main datasource I will use for my program will be the “Recipe Ingredients and Reviews” dataset [5] from Kaggle. The datasource contains a recipes dataset “recpies.csv” of ~13K unique recipes. Each recipe is identified by a unique “RecipeID”. The dataset also shows the ingredients used, cooking instructions, prep time and number of reviews. The main challenge with this dataset is the difficulty of extracting the useful information out of it because the ingredients, amounts and cooking instructions are in raw text format that requires a lot of pre-processing steps. Even though the datasource already contains a clean version of the dataset, I will not be using the clean version because it does not provide the level of details I need for my analysis. Hence, I will do the data cleaning and feature engineering steps myself.

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.003.png)

The datasource also contains a user reviews dataset “reviews.csv” which contains ~1.6M reviews given by 618K users on 74K recipes. In addition, the dataset contains the rating given by each user along with the comment that the user left (if any).

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.004.png)

I noted that the reviews dataset contains reviews on recipes that are not present in the recipes dataset. Hence, for my analysis, I will only keep the reviews given on the recipes in scope of this analysis, amounting to ~1.1M reviews given by 480K users on 6.4K recipes.

A second datasource I will be using is “Recipe Ingredient for Knowledge Mining” dataset [6] from Kaggle. I will use this dataset to train a named entity recognition model to that will help me flag ingredient types and quantities and extract those from the raw text of the ingredients column in “recpies.csv”.



**Methodology**

My goal is to identify novel recipes. To achieve this, I will need to convert each recipe into a feature vector that represents all the information about the recipe. My recipe feature vectors should be created in a way where recipes that are similar to each other should be close to each other in this highly dimensional vector space.


Ingredient features

Keyword features

Time

features


My recipe vector is hence going to be a **concatenation** of 3 different feature vectors:

1. 2 **Time to cook features** (prep time, cook time) which are both readily available in the recipes dataset.

1. N **Ingredient features**, where N is the number of ingredients, each element represents the amount of units of that ingredient used in this recipe (e.g. ‘sugar’: 200 for 200g of sugar). **Named Entity Recognition** (NER) model is used to extract the ingredient names and amounts from the raw ingredient dat.

![Cover image](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.014.png)

1. K **keyword features**, where K is the total number unique keywords that will be extracted from user reviews on the recipes. Each recipe has on average 90 user reviews. I extracted the top 3 keywords (“tags”) of each recipe from the user comments using TF-IDF.

**4.1.	Named Entity Recognition (NER)**

NER is an NLP task concerned with identifying and categorizing key information (entities) in text. NER performs 2 main tasks, step one involves detecting a word or string of words that form an entity and step two involves categorizing the words into predefined entity types. For this project I used the Stanford Stanza NLP library. I decided to work with Stanza because extends the famous Stanford CoreNLP java package and makes it available in python which is more easy to use. Moreover, Stanza achieves state of art performance when compared against start-of-art python packages like Spacy.

![enter image description here](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.015.png)

**Figure 2** - Shows comparison between Stanza and famous NLP Packages

The challenge with my dataset is that the ingredients and their quantities were in raw text format. The default NER model of Stanza was not able to identify ingredients, quantities and units of measurement. Therefore, I had to train my own NER model utilizing stanzas architecture to able to extract Ingredient information. My input to this model was “Recipe Ingredient for Knowledge Mining” dataset. Which contains tokens and their classifications into the 7 main entity classifications (NAME, QUANTITY & UNIT).

![ref2]![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.017.png)

**Figure 3** - Shows the named entities (labels) for tokens 

First, I pre-processed the data to match the Stanford NER trainer requirement that each line of the file is a token followed by a tab and then the entity type. Then I split the 50K tokens data into 75% training and 25% testing and saved the output in TSV (tab separated) format. Now that my data was ready for training I was able to call Stanza’s Conditional Random Field classifier (**CRFClassifier)**[8] and train it to classify my tokens. As opposed to traditional classifiers which predicts a label for a single sample without considering "neighboring" samples, CRFClassifeir can take context into account as it uses word sequences as opposed to just words. Below Is the formulation of CRFClassifier where y is the named entity we are trying to predict it’s probability given an observed x.

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.018.png)

We can see from the problem formulation that the prediction of token at position x relies not only on the label of x but also the label of the t neighboring tokens. Z(X) is only for normalization to convert predictions into probabilities. Maximum likelihood estimate (MLE) can be used to find the optimal weights.

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.019.png)As shown on figure below are examples of sequences of tokens(sentences) that was passed to the classifier and the produced classifications of those tokens into the entity tags (NAME, QUANITITY, UNIT, etc..).

**Figure 4** - Shows how the NER model enabled entity extraction

**4.2	The Recipes ingredient vectors**

While passing the recipes ingredient sequences to the NER model, I also had to clean the recipes data to make sure that quantities like (1 ½) cups are converted into numeric (1.5). In addition, I also removed any punctuation and missing data and Lemmatized the ingredients so that the NER model can perform optimally. 

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.020.png)

Then I processed all the ingredient sequences for all recipes and created the ingredient features dataset at shown below where each feature represents the quantity of ingredient x in recipe R.

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.021.png)

The challenge I had with the ingredient feature vectors is that it was highly dimensional containing 3758 for each of the 12K recipes. To overcome this, I had to come with a novel approach to reduce the dimensionality of my ingredients vectors.

My approach was to use **Lasso regression** to perform feature selection. However, I still had a major challenge where my data was not labelled which was a major blocker against using Lasso. To overcome this, I developed a novel approach, whereby I used **KMeans** algorithm to assign each recipe to a cluster and then I used the cluster labels as my labels for a multi-class lasso classification problem.

First, I started by performing **cross-validation** to pick the ideal number of clusters for KMeans. I knew I was limited to max 10 clusters due to computational power of my laptop, hence I checked for all k clusters from 1-10. The metric I used to measure clustering performance was the Within Cluster Sum of Squares (WCSS) which measures the total squared distances between each point and its cluster center. Below figure shows the WCSS for different number of clusters.

![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\53DFF865.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.022.png)![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\AC58998E.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.023.png)

**Figure 5** - Shows how the WCSS of KMENS for different K values (left) and distribution of recipes within each cluster (right)

After observing the figure 5 above, I opted for k=7 as my number of clusters for KMeans as beyond that the rate of decrease of WCSS is reducing. Figure 5 (right chart) above shows the <a name="_hlk120456362"></a>distribution of recipes within each cluster. Now that I have my data and labels ready, I was able to perform Lasso Regression.

First step was to split the recipes dataframe of size (12k x 3k) data into training and testing randomly with an 80-20% split. I then performed cross-validation to determine the optimal value for regularization parameter λ of the Lasso Regression as per the Lasso regression loss function below.

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.024.png)

My optimal λ was 0.02. Resulting in an accuracy of 84% on the training set and 81% on the testing set. The model evaluation will be discussed further in the evaluation section of this report. As a result of my feature selection approach I was able to reduce the number of features (aka ingredients) needed to represent each recipe by 98%. Hence, my recipe vector was reduced from from R3758 to R75. 

Now, my Recipes feature vectors is a data matrix of size 12,350 recipes by 75 ingredients. Below are examples of ingredients that were kept and ingredients that were neglected.

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.025.png)

**4.2	Recipe keywords vector**

The 2<sup>nd</sup> part of my analysis extracting keywords from user reviews on the recipes. Each recipe has on average 90 user reviews. My idea was to extract the top 3 keywords (“tags”) of each recipe from the user reviews. My hypothesis is that the top keywords representing each recipe can be used to enrich the recipe vector and provide more uniqueness.

First, I started with data cleaning of user reviews which involved:

1. Converting to lowercase
1. RegEx to remove punctuation, links, double spaces, etc..
1. Removing stop words
1. Stemming the words
1. Concatenating all reviews that belong to the same recipe.

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.026.png)


Secondly, I performed Term Frequency-Inverse Document frequency (TF-IDF) analysis on the user reviews data, to find the top keywords for every recipe. TF-IDF is a statistic that can be used to reflect how important a word is to a document in a collection or corpus. The higher the TF-IDF score the more important term t is for doc d. In other words, the higher the TF-IDF score, the more important “key” word t is for recipe d.


|<p>TFIDFt,d= ft,dt'ϵdft',dlogN|{dϵD :tϵd}|</p><p>ft,d →number of times term t mentioned in doc d</p><p>t'ϵdft',d→count of all terms is doc d</p><p>dϵD :tϵd→number of documents containing term t</p><p>N→Total number of documents</p>|
| :- |

Below are some examples of the highest TF-IDF scoring terms for different recipes.

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.027.png)

I performed TF-IDF on my entire dataset of reviews, containing reviews for 6426 recipes, resulting in a TF-IDF matrix of size (6426 x 67417). Again, This feature vector was very high dimensional, and therefore was the need for a novel dimensionality reduction tactic. 

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.028.png)

To achieve this, I utilized Ideas from **Latent Semantic Analysis** **for topic modelling**. The idea is that we can cluster all user reviews into topics. This is achieved using Singular Value Decomposition (SVD) of the TF-IDF vector.

SVDTF-IDF=UΣVT

Where the **U** matrix encodes the topics and represents the assignment of recipes to topics into and the **V** matrix represents the assignment of words to topics. To achieve dimensionality reduction, I performed **truncatedSVD** and was able to achieve 99.3% reduction while still capturing 80% of the variability. Hence, my recipe key words vector was reduced from R67417 to R500. 

![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\9FC106F.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.029.png)

**Figure 6** - Shows the cumulative % of explained variance as number of topics increase

**4.3	Novelty Detection**

To complete my recipe feature vector I concatenating all the 3 vectors as discussed in the intro of the methodology section. After concatenation, my final recipe dataset was now of size m=6426 recipes where each recipe is represented by a vector in R577. 

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.030.png)

After that I performed standardization to put all data on same scale and now it was time to finally apply novelty detection of my final dataset.

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.031.png)

**My hypothesis** for this project is that Novel recipes are “outlier recipes” that are not similar to the majority of recipes and are far from them in the feature space. I experimented with three novelty detection algorithms from 3 different classes namely Isolation Forrest, DBSCAN and Robust covariance estimation. **Robust covariance estimation** is a statistical method for detecting outliers, that relies on density estimations of the datapoints where outliers are points that lie in regions with density < ε. On the other hand **DBSCAN** is a method that tries to group points that are geometrically close (within neighborhood ε) of each other. DBSCAN iteratively groups neighboring points to form clusters. Points that are not within the neighborhood of any cluster are considered outliers. **IsolationForrest** relies on the idea that there is the tendency of anomalous instances in a dataset to be easier to separate from the rest of the points. The algorithm recursively generates partitions on the data by randomly selecting an attribute and then randomly selecting a split value for the attribute. Anomalies will require fewer random partitions to be isolated, compared to normal points and hence can be identified. I was not able to experiment with OneClass-SVM due to processing limitations on my machine.

I passed my recipes dataset through the three models above to compare results. However, since there is no labeled data, I performed t-SNE to plot the recipes on a 2D plot and observe the Novel recipes vs non-novel ones. **t-SNE** (t-distributed Stochastic Neighbor Embedding) is nonlinear dimensionality reduction technique mostly used to understand high-dimensional data and project it into low-dimensional space (like 2D or 3D). t-SNE’s first measures the pairwise similarity between points in the original space. As stated in t-SNE paper [9] similarity of datapoint xⱼ to datapoint xᵢ is the conditional probability Pj|i, that xᵢ would pick xⱼ as its neighbor, which is proportional to probability density under a Gaussian centered at xᵢ and is calculated as below:

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.032.png)

t-SNE then projects the points randomly onto the reduced dimensional plane and again calculates the pairwise similarity qij but this time where xⱼ is centered under a t distribution as it doesn’t have short tail like gaussian and hence prevents crowding of data. 

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.033.png)

Finally, t-SNE’s objective is to reduce the difference between Pj|i and qij as measured by KL-Divergence.

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.034.png)

`	`After applying t-SNE to my recipes dataset I got the below 2D projection: -

![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\8CAB452B.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.035.png)

**Figure 7** - Shows the t-SNE 2D projection of the recipes dataset




Below are the results of anomaly detection with IsolatonForrest and Robust covariance estimation and DBSCAN when visualized with t-SNE

|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\C00B9AE9.tmp]|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\74FECB17.tmp]|
| :- | :- |
|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\DB7F0A0B.tmp]||

DBSCAN does a much better job as it excels in these scenarios by design. The algorithm proceeds by arbitrarily picking up a point in the dataset (until all points have been visited). If there are at least “minPoint” points within a radius of “ε” to the point then we consider all these points to be part of the same cluster. The clusters are then expanded by recursively repeating the neighborhood calculation for each neighboring point. At the end clusters will be formed, but some points will fail to make it to any cluster and those are the outliers**. DBSCAN with ε=3.5 performed the best**.

**4.4	Matrix Completion for recipe recommendation**

Now that I have identified the Novel Recipes in my dataset, as a stretch goal, I wanted to also implement a recipe recommender that looks at the user’s history of recipe ratings [1-5], and only suggests Novel recipes that this user will probably like, as opposed to suggesting any random novel recipe. This is a recommendation system problem that I choose to solve using **Matrix completion** as it takes into account user preferences and preferences of similar users. Matrix completion is essentially a data imputation task where we have a sparse dataset that contains many missing values and would like to fill those missing values using the sparse information present in the matrix. In my case, I have prepared a user reviews matrix which is a sparse (10k users x 6k recipes) matrix of user ratings on recipes. My task is to use matric completion algorithm to fill in or in other words “predict” what the user’s ratings on unseen recipes might be.

Matrix completion algorithm has a fundamental assumption that is that the matrix is low-rank, which means many of the matrix columns are just linear combinations of other columns. Therefore, the matrix completion problem for matrix X becomes as follows:-

min RankZ

s.t. PΩZ=P(X)

X is the original matrix. Z is the new matrix we are trying to compute. P is a projection function where PΩZ=x,   if x is observed0,      if x is missing , so essentially the problem is to find a low rank matrix Z that has the same observed values as X. This is a non-convex problem and hence an appropriate convex relaxation for this problem is the nuclear norm as follows:-

min{Z\*}

s.t. PΩZ=P(X)

The nuclear Norm is the sim of the singular values of a matrix which is a convex problem. One algorithm to solve this problem is **Singular Value Thresholding (SVT).**  The SVT algorithm[10] is as follows:

for a threshold value τ and sequence of positive step sizes δ, Start with Y0=0 ϵRn1xn2 and for k=1,2,3..   itterativly compute below until convergence 

Zk=Sτ(Yk-1) 

Yk=Yk-1+δkPΩ(X-Zk)

St is the singular value thresholding function that works by computing the SVD=UΣVT of Yk and thresholding the singular values by a value of τ then computing Zk=UΣτVT. Ykis then updated in a weighted fashion by the error between Zk and X. This process is repeated for several iterations until convergence. Below is my own python implementation of the above algorithm:

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.039.png)

My algorithm converged is about 1181 iterations as shown below:

![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\DDD26A87.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.040.png)

**Figure 7** - Shows how the error convergence for SVT algorithm

Below is an example prediction for 1 user on multiple recipes and one can observe consistency (similarity) is the recipes:

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.041.png)![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.042.png)




1. **Evaluation and Final Results:**

**5.1 Evaluating the CRFClassfier**

As mentioned in section 4.1, I trained a NER using Stanza’s **CRFClassifier**. Token data was split into 75% training and 25% testing, after training the model to classify the 7 categories (NAME, QUANTITY, SIZE, etc..) it was then tested against the test set. Below are the accuracies, precision, recall, F1 scores, True positives, False Positive and False Negative values broken down by Entity. 

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.043.png)

The model is performing very well in general. The model has a higher precision overall which means it has a higher sensitivity to reducing False positives (aka the number of times the entity is in text but not predicted correctly). However, the model has low number of False Negatives as well (entity was not in text but wrongly predicted). The model performs much stronger on entities like Quantity, SIZE vs entities like TEMP. Overall very satisfactory performance.

**5.1 Evaluating the LassoRegressor**

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.044.png)As mentioned in section 4.2, I used cross validation to train a **Lasso** model to perform feature selection. The labels of this model were the 8 clusters predicted by the KMENs algorithm. Five-fold cross validation was performed over 100 different values for alpha. Optimal alpha was 0.02. Figure 8 below shows the Mean Squared prediction error curve for each fold. Figure 8 on the right shows the coefficient’s decay for different alphas. At alpha 0.02 only 75 coefficients were left.









**Figure 8** - Shows Mean squares prediction error for multiple alpha values (left) and coefficients decay for multiple alphas values (right)

For this Model, I split the data into 80% training on 20% testing. Below figure shows the confusion matrix on my test set for each of the 8 labels:

![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\AE280EC9.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.045.png)

**Figure 9** – confusion matrix of Lasso model

Overall the model had 84% training accuracy and 81% testing accuracy. Which is also very acceptable given that the only purpose of this model is feature selection. From the confusion matrix once can see the model is predicting all classes at a very good rate, however, some classes especially class 1 had more miss-classifications. This could be a sign that this class could be split into 2 or more classes.

**5.3 Evaluating the Novelty Detection Models**

Below are the results of anomaly detection with IsolatonForrest and Robust covariance estimation when visualized with t-SNE projection for different values of contamination factor.

|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\D7A6374D.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.046.png)|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\F8EDB99B.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.047.png)|
| :- | :- |
|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\1C8B93E3.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.048.png)|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\4AA4A161.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.049.png)|
|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\C00B9AE9.tmp]|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\74FECB17.tmp]|
|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\C56665DF.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.050.png)|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\4B51BC3D.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.051.png)|
|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\6F16CA45.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.052.png)|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\C10C9653.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.053.png)|

As observed from above the above comparison as the value of the contamination factor increases, the model flags more recipes as outliers. Contamination factor 0.1 for me seems to the best as it does not include entire clusters of recipes. However, both algorithms seem to only be capturing points at the edges as outliers, whereas I’m looking to captures stand alone recipes that are not part of a cluster, regardless of on the edges or not. I hence experimented with DBSCAN for different values of neighborhood diameter ε and observed the below results:

|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\4E9676D9.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.054.png)|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\729BD74F.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.055.png)|
| :- | :- |
|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\DCA2ED35.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.056.png)|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\DB7F0A0B.tmp]|
|![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\BDFAFB51.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.057.png)||

My Final Model of Choice was DBSCAN with epsilon=3.5.

Now to test if my DBSCAN model is able to capture the outliers, I created 50 fake examples as seen in red in figure[?] below. Th figure on the left shows results After applying DBSCAN with ebs =3.5:

![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\A8D364BF.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.058.png) **![C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\9EA38C25.tmp](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.059.png)** 

As shows above the model still performs very well and was able to capture 82% (41 out of the 50) fake examples that were created. Next, I created a survey showing 6 recipes that were precited by the model as novel and asked for a rating of 1-5 on how Novel they think this recipe is and if they heard about it before. Below are the 7 Recipes that were in the survey:

|<p>garlic chicken and grapes recipe </p><p>Ingredients: 3 tablespoons dijon-style prepared mustard. 3 tablespoons soy sauce. 2 tablespoons honey. 2 tablespoons white wine vinegar. 2 cloves garlic, minced. 2 tablespoons vegetable oil. 3 pounds skinless, boneless chicken breast halves. 1 tablespoon sesame seeds. 2 cups seedless green grapes</p>|
| :- |
|<p>bee sting cake bienenstich ii recipe</p><p>Ingredients: 1 5/8 cups all-purpose flour. 1 tablespoon active dry yeast. 2 tablespoons white sugar. 1 pinch salt. 3/4 cup lukewarm milk. 3 tablespoons butter. 3 tablespoons butter. 1 1/2 tablespoons confectioners sugar. 1 tablespoon milk. 5/8 cup sliced almonds. 1/2 tablespoon cream of tartar. 1 1/2 cups milk. 1/3 cup cornstarch. 1 tablespoon white sugar. 1 egg, beaten. 1 teaspoon almond extract. 1 cup heavy whipping cream. </p>|
|<p>prune and olive chicken recipe </p><p>Ingredients: 3 cloves garlic, minced. 1/3 cup pitted prunes, halved. 8 small green olives. 2 tablespoons capers, with liquid. 2 tablespoons olive oil. 2 tablespoons red wine vinegar. 2 bay leaves. 1 tablespoon dried oregano. salt and pepper to taste. 1 whole chicken, skin removed and cut into pieces. 1/4 cup packed brown sugar. 1/4 cup dry white wine. 1 tablespoon chopped fresh parsley, for garnish</p>|
|<p>whipped carrots and parsnips recipe </p><p>Ingredients: 1 1/2 pounds carrots, coarsely chopped. 2 pounds parsnips, peeled and cut into 1/2 inch pieces. 1/2 cup butter, diced and softened. 1 pinch ground nutmeg. salt to taste. ground black pepper to taste</p>|
|<p>serious herb cheese spread recipe </p><p>Ingredients: 1 package cream cheese, softened. 2 cloves garlic, minced. 1/2 teaspoon prepared mustard. 1/2 teaspoon worcestershire sauce. 1/4 cup chopped parsley. 1/4 cup chopped fresh dill weed. 1/4 cup chopped fresh basil. 1/4 cup chopped black olives. 1 1/2 tablespoons lemon juice</p>|
|<p>roasted root vegetables with apple juice recipe </p><p>Ingredients: 3 tablespoons butter. 3 cups apple juice. 1 cup dry white wine. 1 1/4 pounds turnips. 1 1/4 pounds parsnip. 1 1/4 pounds carrots. 1 1/4 pounds sweet potatoes. 1 1/4 pounds rutabagas. salt and pepper to taste</p>|
|<p>nutty wild rice salad with kiwifruit and red grapes recipe </p><p>Ingredients: 2 1/2 cups chicken stock. 1 cup wild rice. 3 tablespoons lemon juice. 2 teaspoons olive oil. 2 teaspoons honey. 2 kiwis, peeled and diced. 1 cup seedless red grapes, halved. 1 1/2 tablespoons toasted, chopped pecans</p>|

My survey received 23 responses and below is a histogram showing how did survey takers think of the recommended recipes showing that 43.5% gave the recipe 5/5 in novelty and 61% gave them a rating of 4+

![](Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.060.png)



4. **Evaluating the Recommendation system**

To evaluate my matrix completion algorithm, I ran a back test where I hid around 10% of the User rating matrix for the algorithm to predict them, so I can compute the error. Overall the mean absolute error (MAE) for my algorithm was 1.9. Below is the breakdown of the MAE for each of the rating values (1-5).

MAE of 1:- 0.12128146453089245

MAE of 2:- 0.11670480549199085

MAE of 3:- 0.32265446224256294

MAE of 4:- 1.139588100686499

MAE of 5:- 1.9862700228832952

This tells me the model is on average predicting mostly low ratings for recipes and is getting most of the 1 - 3 ratings correct. But doesn’t do as good of a job predicting high rating.

6. **Summary & conclusion**

In this project my objective was to create a recommendation system that recommends Novel Recipes. To achieve this, I engineered a Recipe vector which combines information about the ingredients, their amounts, time to cook and information extracted from user reviews. I trained my own Named entity recognition model to extract ingredient and amount information from the raw text, then performed feature selection in a novel way combining KMeans and Lasso to reduce the dimensionality by 98%. I used TF-IDF to extract key words about recipes from user reviews and then performed truncated SVD to reduce the dimensionality by 99.3%. I then experimented with 3 different Novelty detection techniques, namely, IsolatonForret, EllipticalEnvelop and DBSCAN and visualized the results using t-SNE. DBSCAN with epsilon 3.5 gave me the best results. Finally, I created a collaborative filtering recommendation system using matrix completion and implemented the Singular value thresholding algorithm myself in python to solve it. My Novel Recipe recommender is unlike any recommendation system that I saw before because it balances between recommending novel recipes and recommending relevant recipes at the same time. These same Ideas can be expanded to other domains as well such music, movies images, etc..

**References:**

[1] https://www.bloomberg.com/news/articles/2020-07-07/newly-minted-home-chefs-mark-another-blow-to-u-s-restaurants

[2] https://www.consultancy.uk/news/25412/cooking-at-home-becomes-major-trend-coming-out-of-covid-19 

[3] Jagithyala, Anirudh. “Recommending recipes based on ingredients and user reviews.” 2014. 

[4] Fayyaz, Z, Ebrahimian, M, Nawara, D, Ibrahim, A, Kashef, R. "Recommendation Systems: Algorithms, Challenges, Metrics, and Business Opportunities". Applied Sciences 2020; 10(21):7748. 

[5] https://www.kaggle.com/datasets/kanaryayi/recipe-ingredients-and-reviews?select=reviews.csv

[6] <https://www.kaggle.com/datasets/edwardjross/recipe-ingredient-ner>

[7] Chen, Z., Wang, S. A review on matrix completion for recommender systems. Knowl Inf Syst 64, 1–34 (2022)

[8]<https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ie/crf/CRFClassifier.html>

[9] L. Maaten, G. Hinton. Visualizing Data using t-SNE, 2008

[10] Cai JF, Candès EJ, Shen Z. A singular value thresholding algorithm for matrix completion. SIAM Journal on optimization. 2010;20(4):1956-82.

[ref1]: Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.005.png
[ref2]: Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.016.png
[C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\C00B9AE9.tmp]: Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.036.png
[C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\74FECB17.tmp]: Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.037.png
[C:\Users\ghoneimm\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\DB7F0A0B.tmp]: Aspose.Words.11189391-4874-4e4f-8499-81cd5552cb7b.038.png
