
## 1. New York taxi trips - 2013
<p><img style="float: right;margin:5px 20px 5px 1px; max-width:150px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_496/img/taxi.jpg"> 
<p>In this project, we will analyze a random sample of 49999 New York journeys made in 2013. We will also use regression trees and random forests to build a model that can predict the locations and times when the biggest fares can be earned.</p>
<p>But this is the age of business intelligence and analytics! Even taxi drivers can stand to benefit from some careful investigation of the data, guiding them to maximize their profits</p>


```R
# Loading the tidyverse
library(tidyverse)

# Reading in the taxi data
taxi <- read_csv("datasets/taxi.csv")

head(taxi)
```


## 2. Cleaning the taxi data
<p>The <code>taxi</code> dataset contains the times and price of a large number of taxi trips. Importantly we also get to know the location, the longitude and latitude, where the trip was started.</p>

The data was altered to: <br/>
-Location variables were renamed <br/>
-Journeys with zero fares and tips were removed <br/>
-The log of the sum of fare and tip variables was calculated to deal with outliers, the log was stored in a new variable total

```R
taxi <- taxi %>%
            rename(lat = pickup_latitude, long = pickup_longitude) %>%
            filter(fare_amount > 0, tip_amount > 0) %>%
            mutate(total = log(fare_amount + tip_amount))
head(taxi)
   
```



## 3. Zooming in on Manhattan
<p>While the dataset contains taxi trips from all over New York City, the bulk of the trips are to and from Manhattan, so let's focus only on trips initiated there.</p>

<p>The dataset was filtered to include only trips in Manhattan. The latitude from Manhattan is between 40.70 to 40.83 and longitude is between -74.025 to -73.93.

```R
taxi <- taxi  %>% 
    filter(between(lat, 40.70, 40.83), between(long, -74.025, -73.93))
head(taxi)
```


## 4. Where does the journey begin?
<p>The <code>ggmap</code> package together with <code>ggplot2</code> to visualize where in Manhattan people tend to start their taxi journeys.</p>


```R
# Loading in ggmap and viridis for nice colors
library(ggmap)
library(viridis)

# manhattan <- get_map("manhattan", zoom = 12, color = "bw")
manhattan <- readRDS("datasets/manhattan.rds")

# Drawing a density map with the number of journey start locations
ggmap(manhattan, darken = 0.5) +
   scale_fill_viridis(option = 'plasma') +
   geom_bin2d(data = taxi, aes(x=long, y=lat), bins=60, alpha=0.6) + 
   labs(x='Longitude', y='Latitude', fill='Journeys')
```




![png](output_10_1.png)


## 5. Predicting taxi fares using a tree
<p>The map from the previous task showed that the journeys are highly concentrated in the business and tourist areas. </p>
<p>Regression tree was used to predict the <code>total</code> fare with <code>lat</code> and <code>long</code> being the predictors. The <code>tree</code> algorithm will try to find cutpoints in those predictors that results in the decision tree with the best predictive capability.  </p>


```R
# Loading in the tree package
library(tree)

# Fitting a tree to lat and long
fitted_tree <- tree(total ~lat + long, data = taxi)

# Draw a diagram of the tree structure
plot(fitted_tree)
text(fitted_tree)
```


![png](output_13_0.png)


## 6. Add More predictors.
<p>The tree above looks a bit frugal, it only includes one split: It predicts that trips where <code>lat &lt; 40.7237</code> are more expensive, which makes sense as it is downtown Manhattan. But that's it. It didn't even include <code>long</code> as <code>tree</code> deemed that it didn't improve the predictions. Taxi drivers will need more information than this and any driver paying for your data-driven insights would be disappointed with that.</p>
<p>Some more predictors related to the <em>time</em> the taxi trip was made were added.</p>


```R
library(lubridate)

# Generate the three new time variables
taxi <- taxi %>% 
    mutate(hour = hour(pickup_datetime),
           wday = wday(pickup_datetime, label = TRUE),
           month = month(pickup_datetime, label = TRUE))
head(taxi)
```


## 7. One more tree
<p>Fit a new regression tree where we include the new time variables.</p>


```R
# Fit a tree with total as the outcome and 
# lat, long, hour, wday, and month as predictors
fitted_tree <- tree(total ~ lat + long + hour + wday + month, data = taxi)

# draw a diagram of the tree structure
plot(fitted_tree)
text(fitted_tree)

# Summarize the performance of the tree
print(summary(fitted_tree))
```

    
    Regression tree:
    tree(formula = total ~ lat + long + hour + wday + month, data = taxi)
    Variables actually used in tree construction:
    [1] "lat"
    Number of terminal nodes:  2 
    Residual mean deviance:  0.2676 = 6435 / 24040 
    Distribution of residuals:
        Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    -1.38300 -0.35370 -0.03954  0.00000  0.30620  2.54200 



![png](output_19_1.png)


## 8. One tree is not enough
<p>The regression tree has not changed after including the three time variables. This is likely because latitude is still the most promising first variable to split the data on, and after that split, the other variables are not informative enough to be included. A random forest model, where many different trees are fitted to subsets of the data, may well include the other variables in some of the trees that make it up. </p>


```R
# Loading in the randomForest package
library(randomForest)

# Fitting a random forest
fitted_forest <- randomForest(total ~ lat + long + hour + wday + month, 
                              data = taxi, ntree = 80, sampsize = 10000)

fitted_forest
```


    
    Call:
     randomForest(formula = total ~ lat + long + hour + wday + month,      data = taxi, ntree = 80, sampsize = 10000) 
                   Type of random forest: regression
                         Number of trees: 80
    No. of variables tried at each split: 1
    
              Mean of squared residuals: 0.2641192
                        % Var explained: 2.79




## 9. Plot the predicted fare
<p>In the output of <code>fitted_forest</code> the <code>Mean of squared residuals</code>, that is, the average of the squared errors the model makes. Comparing these numbers, show that <code>fitted_forest</code> has a slightly lower error. Neither predictive model is <em>that</em> good, in statistical terms, they explain only about 3% of the variance. </p>
<p>Now, let's take a look at the predictions of <code>fitted_forest</code> projected back onto Manhattan.</p>


```R
# Extract the prediction from fitted_forest
taxi$pred_total <- fitted_forest$predicted

# Plot the predicted mean trip prices from according to the random forest
ggmap(manhattan, darken = 0.5) +
   scale_fill_viridis(option = 'plasma') +
   stat_summary_2d(data = taxi, aes(x=long, y=lat, z=pred_total), bins=60, alpha=0.6, fun=mean) + 
   labs(x='Longitude', y='Latitude', fill='mean')
```




![png](output_25_1.png)





## 10. Plotting the actual fare
<p>Looking at the map with the predicted fares we see that fares in downtown Manhattan are predicted to be high, while midtown is lower. This map shows the prediction as a function of <code>lat</code> and <code>long</code>, also plot the predictions over time, or a combination of time and space.</p>
<p>For now, let's compare the map with the predicted fares with a new map showing the mean fares according to the data.</p>


## Plot the mean trip prices from the data
```R
ggmap(manhattan, darken = 0.5) +
   scale_fill_viridis(option = 'plasma') +
   geom_bin2d(data = taxi, aes(x=long, y=lat, z=total), bins=60, alpha=0.6, fun=mean_if_enough_data) + 
   labs(x='Longitude', y='Latitude', fill='mean')
```






![png](output_28_2.png)




