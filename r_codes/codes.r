# 1. convert ts object into data.frame
library(zoo)
df_flu = data.frame(month = as.Date(as.yearmon(time(flu))), value = as.matrix(flu))


# 2. convert data.frame to ts object


# 3. find vignettes 
browseVignettes(package = 'packageName')


# 4. 
