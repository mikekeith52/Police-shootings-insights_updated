# Racially Motivated Police Shootings
The question frequently comes up about if when an active police officer shoots and kills an individual, whether that killing was racially motivated.

This is an [update](https://github.com/mikekeith52/Police-shootings-insights) of a descriptive analysis I performed while in graduate school. My goal this time was to expand on the analysis and offer more in-depth insights into *why* the data was saying what it was, as well as try to implement predictive techniques with a Logistic Regression and Random Forest, using a machine learning approach.  

The key takeaways are given their own section [below](#key-findings).  

## Caveats
The data is maintained by the [Washington Post](https://github.com/washingtonpost/data-police-shootings). The owners of this repository offer important insights and caveats about the data. Although this data isn't inclusive of all police killings over the timeframe specified (01/02/2015 to 03/19/2020), it is many times larger than the FBI database that captures the same information. Every one of the 5,174 rows in the dataset represents a death by police shooting. The following information is captured for each observation:

- Name of killed individual
- Date of occurrence
- Manner of Death (shot or shot and tasered)
- If the victim was armed and how (including common and uncommon weapons - guns, crossbows, etc.)
- The age of the individual
- City and state of occurrence
- Whether the official reports indicated if the individual showed signs of mental illness
- The threat level of the individual (attacking, not attacking, undetermined)
- Whether the victim was fleeing
- Whether the story linked to the incident mentioned that the police's body camera was active

This dataset is not an indication of all police altercations, and therefore, some interesting questions cannot be answered with it. Maybe we would like to determine if a given police altercation leads to death, whether race plays a part in being able to predict that outcome. The answer to that question is not available in this dataset.  

In addition to the measures listed above, I added the demographic information from a Wikipedia article to determine the percent of white individuals in the state of every given observation, according to 2012 estimates (last time I used 2000 estimates and argued why I didn't think that would be a problem).  

## Research Question
Given that a police killing has occurred, can we use the factors in this dataset to determine if the individual was more likely white or black? Can we determine how the factors in the dataset move directionally with the outcome of interest?

## Process
I used a systematic Machine Learning approach to analyzing the data. The steps I followed were:
1. [Download data](#download-data)
2. [Clean data for modeling](#clean-data)
3. [Split data into training and testing sets](#split-data)
4. [Train a Logistic Model](#logistic-model)
5. [Train a Random Forest model and tune model hyperparemeters with cross-validation](#random-forest-model)
6. [Visualize results](#visualize-results)

The dependent variable in each model was a binary 0/1 indicator of whether the individual killed in the altercation was white or black (1 for black, 0 for white). I dropped all other races from the dataset.  

I used Python in Jupyter Notebook -- version 3.7.4. Beyond the libraries available in the Anaconda standard library, the following installations are necessary:
- [pydot](https://pypi.org/project/pydot/)
- [tqdm](https://pypi.org/project/tqdm/)

### Download Data
```Python
data = pd.read_csv('https://raw.githubusercontent.com/washingtonpost/data-police-shootings/master/fatal-police-shootings-data.csv')
```
I also used R to scrape a Wikipedia table to obtain statewide racial data. This can be done in Python as well, but some things are easier in R, especially considering that everything comes in dataframe format already. The R code is given below:

```R
setwd('C:/Users/uger7/OneDrive/Documents/PoliceShootings')

library(rvest)
library(stringi)

url <- "https://en.wikipedia.org/wiki/List_of_U.S._states_by_non-Hispanic_white_population"
df1 <- url %>%
  read_html() %>%
  html_nodes(xpath='//*[@id="mw-content-text"]/div/table[1]') %>%
  html_table(fill=T)
df1<-data.frame(df1)

state_info<-data.frame(name=c(state.name,"District of Columbia"),
                       abb=c(state.abb,"DC"))

df1['state_abb'] <- state_info$abb[
  match(df1$'State.Territory',state_info$name)]

df1 <- df1[is.na(df1$state_abb) == F, ]

write.csv(df1,'statewide_race_data.csv',row.names=F)
```

### Clean Data
An important part of this was being able to quickly make all data numeric. Specifically, I wanted all variables to be binary 0/1 type. I used a custom dataframe class, building off the pandas library, to transform the data quickly into this format (my unique code):
``` python
# dynamic dataframe class
class ddf(pd.core.frame.DataFrame):
    """ pandas dataframe with two extra methods
        dummy for creating 0/1 binary variables
        impute_na which uses K Nearest Neighbor to fill in NA values
    """
    def dummy(self,col,exclude_values=[],drop_na=False,exclude_na_levels=False,na_levels=np.NaN,keep_original_col=False,sep=":"):
        """ creates dummy (0/1 variable)
            col is the column you want dummied -- will return as many columns as there are unique values in the column so choose carefully
            exclude_values is a lit of values in that column you want ignored when making dummy columns, you can make this your excluded level
            drop_na is if you want null values droped -- if True, will reduce the number of rows in your dataframe
            exclude_na_levels is if you want your na_levels ignored (meaning no dummy will be made for them)
            na_levels is a user-defined NA level (you can tell it what you want considered null, such as "nan")
            keep_original_col -- if False, the column being dummied will disappear
            sep is how you want the new column names separated (new column name will be the old column name + sep + unique value)
        """
        pd.options.mode.chained_assignment = None
        if drop_na == True:
            self = self.loc[self[col].isnull() == False]
        else:
            self[col].loc[self[col].isnull() == True] = na_levels
            if exclude_na_levels == True:
                exclude_values.append(na_levels)

        self[col] = self[col].astype(str)

        for val in self[col].unique():
            if not val in exclude_values:
                self[f'{col}{sep}{val}'] = 0
                self[f'{col}{sep}{val}'].loc[self[col] == val] = 1

        if keep_original_col == False:
            self.drop(columns=[col],inplace=True)

    def impute_na(self,col,exclude=[]):
        """ uses K-nearest neighbors to fill in missing values
            automatically decides which columns (numerics only) to use as predictors
            a better way to do this is Midas, but this is quick and easy
        """
        predictors=[e for e in self.columns if len(self[e].dropna())==len(self[e])] # predictor columns can have no NAs
        predictors=[e for e in predictors if e != col] # predictor columns cannot be the same as the column to impute (this should be taken care of in the line above, but jic)
        predictors=[e for e in predictors if self[e].dtype in (np.int32,np.int64,np.float32,np.float64,int,float)] # predictor columns must be numeric -- good idea to dummify as many columns as possible
        predictors=[e for e in predictors if e not in exclude] # manually exclude columns (like a dep var)
        clf = KNeighborsClassifier(3, weights='distance')

        df_complete = self.loc[self[col].isnull()==False]
        df_nulls = self.loc[self[col].isnull()]

        trained_model = clf.fit(df_complete[predictors],df_complete[col])
        imputed_values = trained_model.predict(df_nulls[predictors])
        df_nulls[col] = imputed_values

        self[col] = df_complete[col].append(df_nulls[col],ignore_index=False) # preserve index order

    def dummy_regex(self,col,regex_expr=[],sep=':r:'):
        """ this creates a dummy (0/1) variable based on if given phrases (regex_expr) is in the col
            regex_expr should be list type
            sep is how the new column name will be separated
        """
        self[col+sep+'|'.join(regex_expr)] = self[col].astype(str).str.contains('|'.join(regex_expr)).astype(int)
```

This offers three important additional methods for a dataframe:
- dummy() creates a 0/1 for every unique value in a given column, option to drop original column and to exclude values (create omitted classes)
- impute_na() uses a simple k-nearest-neighbors model to fill in missing data; I used this for the age column, which was missing around 50 values (~1.4% of the observations after we subset to White/Black only). I then created buckets for age to create room for error in this process. I did not use the race variable for this imputation as that would create endogeneity in the models
- dummy_regex() will create a dummy variable based on if a word is included in a column

This is how the custom dataframe class was applied to the dataset:

``` Python
# use dynamic dataframe processing
data_processed = ddf(data)
# dummy variables
data_processed.dummy('race',exclude_values=['W'])
data_processed.dummy('gender',exclude_values=['F'])
data_processed.dummy('flee',exclude_na_levels=True,na_levels='nan')
data_processed.dummy('threat_level',exclude_values=['undetermined'])

# dummy variables using regex
data_processed.dummy_regex('armed',['gun'])
data_processed.dummy_regex('armed',['unarmed'])

# impute missing age data 
data_processed.impute_na('age',exclude='race:B')
# bucket age into classes (less than 15, 16 to 25, and 26 to 45; over 45 is the omitted class)
data_processed['child'] = data_processed['age'].apply(lambda x: x <= 15).astype(int)
data_processed['young adult'] = data_processed['age'].apply(lambda x: (x > 15) & (x <= 25)).astype(int)
data_processed['adult'] = data_processed['age'].apply(lambda x: (x > 25) & (x <= 45)).astype(int)

# bucket state demographic info (more than 75% white and 50% to 75% white; less than 50% white is the omitted class)
data_processed['state:almost_all_white'] = data_processed['state_percent_white'].apply(lambda x: x > 75).astype(int)
data_processed['state:majority_white'] = data_processed['state_percent_white'].apply(lambda x: (x >= 50) & (x <= 75)).astype(int)

# now that everyting is a binary 0/1 var, drop all non-integer data types
data_processed.drop(columns=['age','id','state_percent_white'],inplace=True)
for col in data_processed:
    if data_processed[col].dtype not in (np.int32,np.int64,int):
        data_processed.drop(columns=col,inplace=True)
```

To finish the cleaning process, I created seasonal variables (whether the killing occcured in winter, summer, or fall). The final dataset looked like this:

|signs_of_mental_illness|body_camera|race:B|gender:M|flee:Not fleeing|flee:Car|flee:Foot|flee:Other|threat_level:attack|threat_level:other|armed:r:gun|armed:r:unarmed|child|young adult|adult|state:almost_all_white|state:majority_white|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|0|0|1|1|0|0|0|1|0|1|0|0|0|0|1|0|
|1|0|0|1|1|0|0|0|1|0|0|0|0|0|1|0|0|
|0|0|0|1|1|0|0|0|1|0|1|0|0|1|0|0|1|
|0|0|0|1|1|0|0|0|1|0|1|0|0|0|1|1|0|
|0|1|0|0|1|0|0|0|0|1|0|1|0|0|1|1|0|

There were 3,495 observations and 16 predictors to explain one depdendent variable: the race of the individual killed.  

### Split Data
I used an 80/20 split:
```Python
# split 80% of data into training set
X = data_processed.loc[:,data_processed.columns!='race:B']
y = data_processed.loc[:,data_processed.columns=='race:B']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=20)
```

### Logistic Model
The logistic model offers estimated coefficient interpretations. The interpretation of each coefficient's magnitude, which is an odds ratio, can be calculated as its exponentiation minus 1. The direction of the magnitude is the same as its estimated sign, so positive coefficients are indicative of a phenomenon that is more likely to result in a black individual being killed, negative means it's more likely for it to be a white individual. All p-values less than 0.05 can be considered statistically significant at the 95% confidence level.  

```
Optimization terminated successfully.
         Current function value: 0.586618
         Iterations 6
                            Results: Logit
=======================================================================
Model:                Logit              Pseudo R-squared:   0.088     
Dependent Variable:   race:B             AIC:                3314.3656 
Date:                 2020-04-21 12:08   BIC:                3415.2766 
No. Observations:     2796               Log-Likelihood:     -1640.2   
Df Model:             16                 LL-Null:            -1797.8   
Df Residuals:         2779               LLR p-value:        1.7292e-57
Converged:            1.0000             Scale:              1.0000    
No. Iterations:       6.0000                                           
-----------------------------------------------------------------------
                         Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
-----------------------------------------------------------------------
signs_of_mental_illness -0.7134   0.1109 -6.4317 0.0000 -0.9308 -0.4960
body_camera              0.5758   0.1280  4.4982 0.0000  0.3249  0.8268
gender:M                 0.2792   0.2150  1.2988 0.1940 -0.1421  0.7006
flee:Not fleeing        -0.1273   0.2252 -0.5653 0.5719 -0.5687  0.3141
flee:Car                -0.1272   0.2411 -0.5276 0.5978 -0.5997  0.3453
flee:Foot                0.3898   0.2449  1.5917 0.1114 -0.0902  0.8699
flee:Other              -0.5281   0.3279 -1.6107 0.1072 -1.1708  0.1145
threat_level:attack     -0.0241   0.2147 -0.1123 0.9106 -0.4449  0.3967
threat_level:other      -0.1208   0.2181 -0.5540 0.5796 -0.5482  0.3066
armed:r:gun              0.0787   0.0992  0.7931 0.4277 -0.1157  0.2730
armed:r:unarmed          0.2172   0.1754  1.2385 0.2155 -0.1265  0.5609
child                    1.3898   0.6605  2.1041 0.0354  0.0952  2.6843
young adult              1.5775   0.1348 11.7050 0.0000  1.3133  1.8416
adult                    0.8587   0.1154  7.4433 0.0000  0.6326  1.0848
state:almost_all_white  -0.4268   0.1285 -3.3205 0.0009 -0.6788 -0.1749
state:majority_white     0.0698   0.1137  0.6137 0.5394 -0.1531  0.2927
intercept               -1.5374   0.3653 -4.2085 0.0000 -2.2535 -0.8214
=======================================================================
```

As suspected, none of the seasonal variables are statistically significant at any level. The statistically significant variables are:
- signs_of_mental_illness
- body_camera
- child
- young adult
- adult
- state:almost_all_white

The last time I ran this exercise with a smaller dataset, I used slightly different model inputs, and the significant variables were:
- The white percentage in the given state of the shootings
- The age of the suspect
- Whether the suspect displayed signs of mental illness
- Whether the officer was wearing an active body camera
- The gender of the suspect
- Whether or not the suspect was carrying a "toy weapon" 

So, almost all the same variables were evaluated as statistically significant this time. I combined the Hispanic and Black races last time I did this. This time, I was only interested in white vs. black. Further exploration of these model interpretations can be found in the [Key Findings](#key-findings) section.  

The total accuracy of this model, when tested on the test split was 70%. The no-information rate was 64%. This model is slightly better than simply guessing, which to me, can be considered a success since these inputs aren't all inclusive into poential factors that affect the outcome, nor is this an easy question to answer by any measure.

```
No Information Rate: 0.64
Model Total Accruacy: 0.70
```

The precision and recall measures can be viewed as well:

```
              precision    recall  f1-score   support

           0       0.71      0.90      0.79       448
           1       0.65      0.34      0.45       251

    accuracy                           0.70       699
   macro avg       0.68      0.62      0.62       699
weighted avg       0.69      0.70      0.67       699
```

### Random Forest Model
Unlike a Logistic Model, a Random Forest model can have many different hyperparemeter values to test. With a Logistic Regression, you can tune the cutoff level to round the predict outcome on and you can test the number of inputs to include the model. I chose not to tune the Logistic Regression as preliminary attempts to do this indicated to me that there was no way to significantly improve the base model. However, with the Random Forest, I chose to use 3-fold cross-validation to tune five different hyperparameters (my unique code, except the expand_grid function which I obtained from [Stack Overflow](https://stackoverflow.com/questions/12130883/r-expand-grid-function-in-python)):
``` Python
# Random Forest Classifier
hyper_params = {
    'max_depth':[10,20],
    'n_estimators':[100,500,1000],
    'min_samples_split':[2,4,6],
    'max_features':['auto','sqrt'],
    'max_samples':[0.5,.99]
}

# expand grid to get all possible combos
def expand_grid(dictionary):
    """ takes a dictionary of lists, and expands out arrays into a pandas dataframe
    """
    return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())
grid = expand_grid(hyper_params)

# view result shape
grid.shape

# create cross validate function
def cross_validate_rf(X_train,y_train,k=3,grid=grid,loss='gini'):
    """ cross validates across k folds
        uses non-sepcified random states, so results can change each time
        returns two outputs: the optimal hyperparameters and the full grid with derived error metrics
    """
    # copy our grid to write error metrics into it
    hyper_grid = grid.copy()
    
    # create the error columns for each cross-validation fold
    for i in range(1,k+1):
        hyper_grid[f'error_{i}'] = 0
    
    # run the random forest estimator through the gird of parameters and score each cv fold
    for i, row in tqdm(hyper_grid.iterrows()):
        rf = RandomForestClassifier(
            n_estimators = row['n_estimators'],
            max_depth = row['max_depth'],
            min_samples_split = row['min_samples_split'],
            max_features = row['max_features'],
            max_samples = row['max_samples']
        )
        errors = 1 - cross_val_score(rf, X_train, y_train, cv=k, scoring = 'accuracy')
        # write each cv score to its own column
        for idx, e in enumerate(errors):
            hyper_grid.loc[hyper_grid.index==i,f'error_{idx+1}'] = e
    # take the mean of each cross-validated iteration to get the final error metric for each line of hyper parameters
    hyper_grid['total_error'] = hyper_grid[[e for e in hyper_grid.columns if e.startswith('error_')]].mean(axis=1)
    min_error = hyper_grid['total_error'].min()
    # return the row with the lowest error metric as well as the full set of results
    return hyper_grid.loc[hyper_grid['total_error'] == min_error], hyper_grid
```
The optimal hyperparameters were returned:

|max_depth|n_estimators|min_samples_split|max_features|max_samples|error_1|error_2|error_3|total_error|
|---|---|---|---|---|---|---|---|---|
|10|1000|6|sqrt|0.5|0.330472|0.321888|0.332618|0.328326|

The accuracy of the model on the test dataset was also greater than the no-information rate, but worse than the Logistic Model:
```
No information rate: 0.64
Total accuracy of model: 0.68
```

The precision and recall measures:

```
              precision    recall  f1-score   support

           0       0.70      0.88      0.78       448
           1       0.61      0.33      0.42       251

    accuracy                           0.68       699
   macro avg       0.65      0.60      0.60       699
weighted avg       0.67      0.68      0.65       699
```

Without hyper-parameter tuning and using the default parameters from the RandomForestClassifier from sklearn, the total accuracy was 0.65.

### Visualize Results
The ROC Curve from the Logistic Model is bumpy, but the curve expands out from y=x, another metric of our model's adequate efficiency in making predictions:

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/Log_ROC.png)

Feature importance from the Random Forest model:

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/feature_importance.png)

It is interesting to select one of the bagged trees from the Random Forest model and it gives some insight into which variables are influencing the algorithms's decisions in certain directions.

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/tree.png)

This is easier to see with a smaller-tree subset.

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/small_tree.png)

All code to produce these visualizations included in predict_shootings.ipynb.

## Key Findings

### Race in States
In 2018 when I completed this analysis, I was interested to know that despite the fact that media seems to cover the killings of African Americans more, whites are actually killed more often by police. However, when you control for national averages, you can clearly see that whites are under-represented in this data and blacks are highly over-represented.

|Race|Number times in dataset|Percent of Total in Data|Natl Avg|Ratio: Perc in Data to Natl Avg|
|---|---|---|---|---|
|Asian|54|1.67%|4.80%|34.74%|
|Black|825|25.48%|12.60%|202.21%|
|Hispanic|592|18.28%|16.30%|112.16%|
|Native Amer|57|1.76%|0.90%|195.59%|
|Other|34|1.05%|9.30%|11.29%|
|White|1676|51.76%|72.40%|71.49%|

The black race is represented in this dataset more than twice as much as you would expect if she were looking purely at demographic information with no racial presuppositions, and whites are only represented 71% as much. Through March of 2020, this table looks like this (using the same national demographic averages as before):

|Race|Number times in dataset|Percent of Total in Data|Natl Avg|Ratio: Perc in Data to Natl Avg|
|---|---|---|---|---|
|Asian|86|1.89%|4.80%|39.38%|
|Black|1210|26.59%|12.60%|211.03%|
|Hispanic|848|18.63%|16.30%|114.30%|
|Native Amer|75|1.65%|0.90%|183.33%|
|Other|46|1.01%|9.30%|10.86%|
|White|2286|50.23%|72.40%|69.38%|

It doesn't look like it has changed much since September of 2018 (the proportion of blacks is up slightly and the proportion of whites is down), but the question is, has it statistically? Are the proportion of blacks being killed by the police really increasing over time? Or has it stayed the same? We can perform a chi-squared test to figure this out. The chi-squared test will use our observed number for each group and compare it with the number we'd expect had the percentages remained exactly even since 2018 (because of rounding, some of these calculations may appear off, but they are technically correct).

|Race|Number times in dataset|Expected number of times in dataset to match 2018|Difference|Difference^2|X^2_race_i|
|---|---|---|---|---|---|
|Asian|86|76|10|100|1.32|
|Black|1210|1160|50|2541|2.19|
|Hispanic|848|832|16|258|0.31|
|Native Amer|75|80|-5|26|0.32|
|Other|46|48|-2|3|0.07|
|White|2286|2356|-70|4844|2.06|

This gives us a total chi-squared coefficient value of 6.26. The critical value to be 95% certain with 5 degrees of freedom that the population is changing over time is 11.07. Therefore, we cannot reject the null hypothesis and do not have enough evidence to conclude that the population has changed over time.  

An interesting table to look at is to see which states are the most likely and least likely to have more blacks killed in altercations than whites given their own demographic make-ups. I did this by matching the percentage white each state was, according to [Wikipedia](https://en.wikipedia.org/wiki/List_of_U.S._states_by_non-Hispanic_white_population), divided by the percentage of whites that appeared in the dataset for that given state. The top 5 states that were most likely to see blacks killed were:

|State Name|B|W|killed percent white|in state percent white|ratio killed vs. in state|
|---|---|---|---|---|---|
|District of Columbia|12|1|7.7%|35.3%|21.8%|
|Rhode Island|2|1|33.3%|75.4%|44.2%|
|Illinois|57|28|32.9%|62.9%|52.4%|
|Louisiana|56|36|39.1%|59.7%|65.5%|
|Maryland|43|24|35.8%|53.8%|66.6%|

And the top 5 most likely to see whites killed were:

|State Name|B|W|killed percent white|in state percent white|ratio killed vs. in state|
|---|---|---|---|---|---|
|Texas|90|162|64.3%|44.3%|145.1%|
|Arizona|16|104|86.7%|57.0%|152.3%|
|California|118|203|63.2%|39.2%|161.3%|
|New Mexico|1|24|96.0%|39.7%|241.8%|
|Hawaii|1|3|75.0%|22.8%|329.0%|

Sample size can play a factor into these results (I'm not complaining!). It is interesting data nonetheless. The full dataset is available as states_ratio.csv.  

One of my key findings the first time I completed this analysis (9/12/2018) was that as "the percentage of non-white people increases per state, the proportion of blacks, Hispanics, Native Americans, and 'other race' killed by police increases almost exactly proportionally" all else held constant. This was dervived from the interpretation of an applied Logistic Regression model. One may think the last two displayed talbes dispute this finding, as we can clearly see that states are very different in this regard. However, interpretations of multivariate models can be different than a purely univariate view of the data, so the original conclusion still stands when all other factors in the data are controlled for.  

### Age of Killed Individual
The first analysis I completed suggested that younger individuals were more likely to be black, and this is backed up in every view of the data I've examined. Particularly, black young adults (between 16 and 25) are much more likely to be killed by police.

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/age_histogram.png)

A recent finding from the [American Psychological Association](https://www.apa.org/news/press/releases/2014/03/black-boys-older) states that "Black boys as young as 10 may not be viewed in the same light of childhood innocence as their white peers, but are instead more likely to be mistaken as older, be perceived as guilty and face police violence if accused of a crime." The data does not refute this. In the Random Forest model, the young adult variable was most helpful in making predictions. In the Logistic Model, the same variable was the most statistically significant (and its sign indicated that young adults killed are much more likely to be black). All three age variables indicated that younger individuals killed by the police are significantly more likely to be African American. It is actually really jarring and should be paid attention to.  

I can already see the argument against this--younger blacks, on average, are more threatening in altercations, more likely to be armed, more likely to provoke the police, etc. It's not racist, it's just the truth; it might not even be their fault. It might just the environment they were raised in. Whatever the reason, it's the truth.  

But, remember, many of those extraneous factors are controlled for in the results. We have a variable for threat level. We have a variable describing how the individual was or was not armed and another for whether he was fleeing. If blacks or whites were doing these actions more often than the other when killed by police, the data would show it. But none of these variables are statistically significant. Instead, the data suggests that *just the fact of a black male being between 16 and 25 years of age is in itself an indicator of an increased ability of our law enforcement to execute that individual*, all other factors being held equal. Indeed, it is the most determinant factor. That is an incredible finding.  

### Signs of Mental Illness
In both rounds of analysis, the signs_of_mental_illness variable was highly indicative of an individual being more likely white when killed by police. I have thought about this insight and suspect the answer may be that we are more willing generally as a society to label whites who misbehave as mentally ill. There is [research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4318286/) that backs this up. When a non-white person commits a crime, that person is more likely to just be "bad." With whites, we are more likely, when we are reporting and talking about this, to think there must have been some uncontrollable factor, like mental illness, that caused the individual to get in a situation to be killed by police. Again, just like the way we view younger African Americans as older and more threatening than their white peers, it is likely that the signs_of_mental_illness variable in this dataset points to a larger, societal bias.  

### Body camera
In both rounds of analyses, the data suggests that officers with active body cameras are more likely to have killed a black individual, and this is a highly significant input. I've spent a little bit of time thinking about this, and I believe the most likely explanation is that this is reflective of how news is consumed in America. The Washington Post documentation suggests that this variable is assigned a True value only when the active body camera is mentioned in the associated story of the police killing. It is not actually explicitly determined by whether the officer was wearing an active body camera. What this suggests to me is that in news stories about blacks being killed, the question of whether the officer was wearing a body camera is more likely to come up. These types of killings are viewed as more scandalous in our society than when a white person is killed (usually) and one of the first things we want to know, no matter what side of the political aisle we're on, is if we can see the camera to decide for ourselves whether the killing was justified or not. This makes more sense to me than the body camera coming into the decision-making of the officer to kill a black or white individual to such a large degree.  

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/body_camera_boxplot.png)
