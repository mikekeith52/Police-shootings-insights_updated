

# Racially Motivated Police Shootings
The question frequently comes up about if when the police shoot and kill an individual, whether that killing was racially motivated.

## Update
This project is an [update](https://github.com/mikekeith52/Police-shootings-insights) of one I completed a few years ago with the same data. My initial thought was that the predictors in this dataset didn't have any strong predictive power, so the last project I completed was fully descriptive in nature. I wanted to know the direction of the factors in the dataset and whether we could see statistical significance by measuring their effect on the target variable. I wasn't interested in knowing if the variables could demonstrate any predictive power.  

This time, I wanted to expand on the descriptive analysis, offering more in-depth insights into *why* the data was saying what it was, as well as try to offer predictive techniques that could out-perform simply guessing about the data.  

Programming note: last time I used R, this time I used Python.  

The key takeaways are given their own section [below](#key-findings). Before that, I will review the data in question and my methodology in deriving the conclusions.  

## Caveats
The [Washington Post GitHub page](https://github.com/mikekeith52/data-police-shootings) offers important insights and caveats about the data. Although this data isn't inclusive of all police killings over the timeframe (01/02/2015 to 03/19/2020), it is many times larger than the FBI database that captures the same information. Every one of the 5,174 rows in the dataset represents a death by police shooting. The dataset includes measures of:

- Name of killed individual
- Date of occurrence
- Manner of Death (shot or shot and tasered)
- If the victim was armed and how (including common and uncommon weapons - guns, crossbows, etc.)
- The age of the individual
- City and state of occurrence
- Whether the police reports indicated if the individual showed signs of mental illness
- The threat level of the individual (attacking, not attacking, undetermined)
- Whether the victim was fleeing
- Whether the story linked to the incident mentioned that the police's body camera was playing

This dataset is not an indication of all police altercations. An interesting research question would be to predict if a given police altercation leads to death and whether race plays a part in being able to predict that outcome. The answer to that question is not available in this dataset.  

In addition to the measures listed above, I added the demographic information from a Wikipedia site to determine the percent of white individuals in the state for every given observation, according to 2012 estimates (last time I used 2000 estimates and argued why I didn't think that would be a problem). I also added seasons - winter, fall, and summer (where spring is omitted) to see if that made any difference; I didn't think it would.  

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

The dependent variable in each model was a binary 0/1 indicator of whether the individual killed in the altercation was white or black (1 for black, 0 for white).  

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
An important part of this was being able to quickly make all data numeric, specifically I wanted all variables to binary 0/1. I used a custom dataframe class, building off the pandas library, to accomplish this quickly (my unique solution):
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
- impute_na() uses a simple k-nearest-neighbors model to fill in missing data; I used this for the age column, which was missing around 50 values, I then created buckets for age to create room for error in this process. I did not use the race variable for this imputation as that would create endogeneity in the models
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

### Split Data
I used an 80/20 split:
```Python
# split 80% of data into training set
X = data_processed.loc[:,data_processed.columns!='race:B']
y = data_processed.loc[:,data_processed.columns=='race:B']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=20)
```

### Logistic Model

The logistic model offers estimated coefficient interpretations. The interpretation of each coefficient's magnitude, which is an odd-ratio, is its exponentiation minus 1. The direction of the magnitude is the same as its estimated sign, so positive coefficients are indicative of a phenomenon that is more likely to result in a black individual being killed, negative means it's more likely for it to be a white individual. All p-values less than 0.05 can be considered statistically significant with 95% certainty.

```
Optimization terminated successfully.
         Current function value: 0.586120
         Iterations 6
                            Results: Logit
=======================================================================
Model:                Logit              Pseudo R-squared:   0.088     
Dependent Variable:   race:B             AIC:                3317.5804 
Date:                 2020-04-15 12:55   BIC:                3436.2993 
No. Observations:     2796               Log-Likelihood:     -1638.8   
Df Model:             19                 LL-Null:            -1797.8   
Df Residuals:         2776               LLR p-value:        3.9079e-56
Converged:            1.0000             Scale:              1.0000    
No. Iterations:       6.0000                                           
-----------------------------------------------------------------------
                         Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
-----------------------------------------------------------------------
signs_of_mental_illness -0.7156   0.1110 -6.4453 0.0000 -0.9332 -0.4980
body_camera              0.5833   0.1282  4.5489 0.0000  0.3320  0.8346
summer                   0.0631   0.1201  0.5251 0.5995 -0.1723  0.2984
fall                    -0.0373   0.1237 -0.3011 0.7634 -0.2798  0.2053
winter                   0.1481   0.1180  1.2554 0.2093 -0.0831  0.3793
gender:M                 0.2779   0.2153  1.2908 0.1968 -0.1440  0.6998
flee:Not fleeing        -0.1138   0.2255 -0.5047 0.6138 -0.5557  0.3281
flee:Car                -0.1194   0.2412 -0.4949 0.6206 -0.5922  0.3534
flee:Foot                0.3971   0.2450  1.6209 0.1050 -0.0831  0.8773
flee:Other              -0.5120   0.3286 -1.5582 0.1192 -1.1559  0.1320
threat_level:attack     -0.0327   0.2156 -0.1514 0.8796 -0.4553  0.3900
threat_level:other      -0.1343   0.2193 -0.6126 0.5401 -0.5641  0.2955
armed:r:gun              0.0780   0.0993  0.7855 0.4321 -0.1166  0.2727
armed:r:unarmed          0.2198   0.1754  1.2533 0.2101 -0.1240  0.5636
child                    1.3955   0.6588  2.1182 0.0342  0.1043  2.6868
young adult              1.5771   0.1349 11.6878 0.0000  1.3126  1.8415
adult                    0.8610   0.1155  7.4532 0.0000  0.6346  1.0874
state:almost_all_white  -0.4263   0.1286 -3.3143 0.0009 -0.6785 -0.1742
state:majority_white     0.0741   0.1139  0.6503 0.5155 -0.1491  0.2972
intercept               -1.5889   0.3733 -4.2564 0.0000 -2.3206 -0.8573
=======================================================================
```

As suspected, none of the seasonal variables are statistically significant at any level. The statistically significant variables are:
- signs_of_mental_illness
- body_camera
- flee:Foot
- child
- young adult
- adult
- state:almost_all_white

The last time I ran this exercise with a smaller dataset, I used slightly different model inputs, and the significant inputs were:
- The white percentage in the given state of the shootings
- The age of the suspect
- Whether the suspect displayed signs of mental illness
- Whether the officer was wearing an active body camera
- The gender of the suspect
- Whether or not the suspect was carrying a "toy weapon" 

So, the same variables came up statistically significant this time, except "toy weapon," which I omitted this time. I combined the Hispanic and Black races last time I did this. This time, I was only interested in white vs. black. Further exploration of these model interpretations can be found in the [Key Findings](#key-findings) section.  

The total accuracy of this model, when tested on the test split was 70%. The no-information rate was 64%. This model is slightly better than simply guessing, which to me, is better-than-expected considering these aren't the best inputs, nor is this the easiest question to answer.

```
No Information Rate: 0.64
Model Total Accruacy: 0.70
```

### Random Forest Model
A Random Forest model can be influenced by its hyperparameters much more than a Logistic Model. I chose to use 3-fold cross-validation, on the training set only, to tune the RF model (my unique code, except the expand_grid function which I obtained from [Stack Overflow](https://stackoverflow.com/questions/12130883/r-expand-grid-function-in-python)):
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

The accuracy of the model on the test dataset was also greater than the no-information rate, but worse than the Logistic Model
```
No information rate: 0.64
Total accuracy of model: 0.68
```

Without hyper-parameter tuning and using the default parameters from the RandomForestClassifier from sklearn, the total accuracy is 0.65.

### Visualize Results
The ROC Curve from the Logistic Model is bumpy, but the curve expands out from y=x, another metric of our model's adequate efficiency in making predictions:

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/Log_ROC.png)

Feature importance from the Random Forest model:

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/feature_importance.png)

It is interesting to select one of the bagged trees from the Random Forest model and it gives some insight into which variables are influencing the algorithms's decisions in certain directions.

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/tree.png)

This is easier to see with a smaller-tree subset.

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/small_tree.png)

## Key Findings

### Race in States
One of my key findings the first time I completed this analysis (9/12/2018) was that as "the percentage of non-white people increases per state, the proportion of blacks, Hispanics, Native Americans, and 'other race' killed by police increases almost exactly proportionally." One of the most jarring tables from this analysis was the one below:

|Race|Number times in dataset|Percent of Total in Data|Natl Avg|Ratio: Perc in Data to Natl Avg|
|---|---|---|---|---|
|Asian|54|1.67%|4.80%|34.74%|
|Black|825|25.48%|12.60%|202.21%|
|Hispanic|592|18.28%|16.30%|112.16%|
|Native Amer|57|1.76%|0.90%|195.59%|
|Other|34|1.05%|9.30%|11.29%|
|White|1676|51.76%|72.40%|71.49%|

The finding being that blacks are represented in this dataset more than twice as much as you would expect if you were looking purely at demographic information with no racial presuppositions, and whites are only represented 71% as much. In the updated dataset, this looks like the following (using the same national demographic averages from 2010):

|Race|Number times in dataset|Percent of Total in Data|Natl Avg|Ratio: Perc in Data to Natl Avg|
|---|---|---|---|---|
|Asian|86|1.89%|4.80%|39.38%|
|Black|1210|26.59%|12.60%|211.03%|
|Hispanic|848|18.63%|16.30%|114.30%|
|Native Amer|75|1.65%|0.90%|183.33%|
|Other|46|1.01%|9.30%|10.86%|
|White|2286|50.23%|72.40%|69.38%|

You may notice these numbers do not add up to the total (5,174) stated earlier. The remaining observations belong to those where the racial data is missing in the dataset. Reading the Washington Post documentation, these appear to be cases where the race was not disclosed to the public. Because there is no systematic way to assign these observations, they are removed from the dataset.  

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

### Age of Killed Individual
The first analysis I completed suggested that younger individuals were more likely to be black, and this is backed up in every view of the data. Particularly, black young adults (between 16 and 25) are much more likely to be killed by police.

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/age_histogram.png)

A lot of people, social scientists espcially, have speculated that in our country, many white people, including police officers, view black men and women as more developed at younger ages, more threatening, and more responsible for their wrong actions than whites at similar ages. The data does not refute the opinions of people who assert this. In the Random Forest model, the young_adult variable was most helpful in making predictions. In the Logistic Model, the same variable was the most statistically significant (and its sign indicated that young_adults killed are much more likely to be black). All three age variables indicated that younger individuals killed by the police are significantly less likely to be white. It is actually really jarring and should be paid attention to. When one is making judgements of how old and/or threatening an individual appears, she should call into question her own racial biases before making any conclusions in that regard.  

I can already see the argument against this--younger blacks, on average, are more threatening in altercations, more likely to be armed, more likely to provoke the police, etc. It's not racist, it's just the truth; it might not even be their fault. It might just the environment they were raised in. Whatever the reason, it's the truth.  

But, remember, many of those extraneous factors are controlled for in the results. We have a variable for threat level. We have a variable describing how the individual was or was not armed and another for whether they were fleeing. If blacks or whites were doing these actions more often than the other when killed by police, the data would show it. But none of these variables are statistically significant. Instead, the data suggests that *just the fact of a black male being between 16 and 25 years of age is in itself an indicator of an increased ability of our law enforcement to execute that individual*, all other factors being held equal. Indeed, it is the most determinant factor. That should be shocking to everybody reading.  

### Signs of Mental Illness
In both rounds of analysis, the signs_of_mental_illness variable was highly indicative of an individual being more likely white when killed by police. I have thought about this finding and suspect the answer may be that we are more willing generally as a society to label whites who misbehave as mentally ill. When a non-white person commits a crime, that person is more likely to just be "bad." With whites, we are more likely, when we are reporting and talking about this, to think there must have been some uncontrollable factor, like mental illness, that caused the white individual to get in a situation to be killed by police. Again, it's likely a sign of implicit bias.  

### Body camera
In both rounds of analyses, the data suggests that officers with active body cameras are more likely to have killed a black individual, and this is a highly significant input. I've spent a little bit of time thinking about this, and I believe the most likely explanation is this is reflective of how news is consumed in America. The Washington Post documentation suggests that this variable is assigned a True value only when the active body camera is mentioned in the associated story of the police killing. It is not actually explicitly determined by whether the officer was wearing an active body camera. What this suggests to me is that in news stories about blacks being killed, the question of whether the officer was wearing a body camera is more likely to come up. These types of killings are viewed as more scandalous in our society than when a white person is killed (usually) and one of the first things we want to know, no matter what side of the political aisle we're on, is if we can see the camera to decide for ourselves whether the killing was justified or not. This makes more sense to me than the body camera coming into the decision-making of the officer to kill a black or white individual to such a large degree.  

![](https://github.com/mikekeith52/Police-shootings-insights_updated/blob/master/img/body_camera_boxplot.png)

### Seasons 
Seasons (winter vs. fall vs. spring vs. summer) appear to play a negligible role in predicting the outcome of interest in this dataset. That is not unsurprising, although it should be noted that the Random Forest model did not find these variables completely unhelpful.