# 0. Misc
## abort an expression: press ESC key
## show the results of a new assignment: wrap it with ()
##


# 1. convert ts object into data.frame
library(zoo)
ts_to_df <- function(ts) {
    data.frame(month = as.Date(as.yearmon(time(ts))), value = as.matrix(ts))
}


# 2. convert data.frame to ts object
df <- df[order(df$Date), ]
df.ts <- ts(df$value, start=c(year(min(df$Date)), month(min(df$Date)), end=c(year(max(df$Date)), month(max(df$Date))), frequency=12)


# 3. find vignettes 
browseVignettes(package = 'packageName')


# 4. create a new virtual environment with Packrat
# a) create a new R project with a new directory
# b) packrat::init("~/projects/babynames")
# c)


# 5. show all install packages
ip = as.data.frame(installed.packages()[,c(1,3:4)])
ip = ip[is.na(ip$Priority),1:2,drop=FALSE]
ip


# 6. show the full tibble
tibble.name %>% print(n = Inf)
View(tibble.name)


# 7. ggplot: graphics grammar
ggplot(data = <DATA>) + 
  <GEOM_FUNCTION>(
     mapping = aes(<MAPPINGS>),
     stat = <STAT>, 
     position = <POSITION>
  ) +
  <COORDINATE_FUNCTION> +
  <FACET_FUNCTION>


# 8. show all historical commands on an object
type object name, press cmd + up-arrow


# 9. ggplot: show a scatter plot of x, y and a smoothed line
tibble.name %>% 
  ggplot(mapping = aes(x = x, y = y)) +
    geom_point() + 
    geom_smooth(se = FALSE)


# 10. count the rows by a variable
diamonds %>% count(cut)
diamonds %>% group_by(cut) %>% summarise(n())


# 11. ggplot: compare density of different variables (area under each histogram is equal to 1 for fair comparison)
ggplot(data = diamonds, mapping = aes(x = price, y = ..density..)) + 
  geom_freqpoly(mapping = aes(colour = cut), binwidth = 500)


# 12. ggplot: create heatmap by count
diamonds %>% 
  count(color, cut) %>%  
  ggplot(mapping = aes(x = color, y = cut)) +
    geom_tile(mapping = aes(fill = n))


# 13. Prevent the conversion from string to factors
x.df <- data.frame(xNum, xLog, xChar, stringsAsFactors = FALSE)


# 14. Remove all variables
rm(list=ls())


# 15. Check random subset of data frame
library(car)
some(df.name)


# 16. Describe a dataframe's numeric fields
library(psych)
describe(df.name)


# 17. Plot a histogram
hist(store.df$p1sales, 
     main="Product 1 Weekly Sales Frequencies, All Stores",
     xlab="Product 1 Sales (Units)",
     ylab="Relative frequency",
     breaks=30, 
     col="lightblue", 
     freq=FALSE,                            # freq=FALSE means to plot density, not counts
     xaxt="n")                              # xaxt="n" means "x axis tick marks == no"

axis(side=1, at=seq(60, 300, by=20))        # add the x axis (side=1) tick marks we want


lines(density(store.df$p1sales, bw=10),    # "bw= ..." adjusts the smoothing
      type="l", col="darkred", lwd=2)      # lwd = line width



# 18. Box plots
## one group:
boxplot(store.df$p2sales, xlab="Weekly sales", ylab="P2",
        main="Weekly sales of P2, All stores", horizontal=TRUE)

## multiple groups:
boxplot(p2sales ~ p2prom, data=store.df, horizontal=TRUE, yaxt="n", 
     ylab="P2 promoted in store?", xlab="Weekly sales",
     main="Weekly sales of P2 with and without promotion")
axis(side=2, at=c(1,2), labels=c("No", "Yes"))


# 19. Plot Empirical CDF
plot(ecdf(store.df$p1sales),
     main="Cumulative distribution of P1 Weekly Sales",
     ylab="Cumulative Proportion",
     xlab=c("P1 weekly sales, all stores", "90% of weeks sold <= 171 units"),
     yaxt="n")
axis(side=2, at=seq(0, 1, by=0.1), las=1, 
     labels=paste(seq(0,100,by=10), "%", sep=""))
# add lines for 90%
abline(h=0.9, lty=3)
abline(v=quantile(store.df$p1sales, pr=0.9), lty=3)


# 20. Use aggregate function
p1sales.sum <- aggregate(store.df$p1sales, 
                         by=list(country=store.df$country), sum)


# 21. Plot a scatter plot
plot(cust.df$age, cust.df$credit.score, 
     col="blue",
     xlim=c(15, 55), ylim=c(500, 900), 
     main="Active Customers as of June 2014",
     xlab="Customer Age (years)", ylab="Customer Credit Score ")
abline(h=mean(cust.df$credit.score), col="dark blue", lty="dotted")
abline(v=mean(cust.df$age), col="dark blue", lty="dotted") ## add straignt lines for slope/intercept
point() ## add specific points
lines() ## add a set of lines
legend() ## add a legend


# 22. Plot a scatterplot matrix
library(car)   # install if needed
scatterplotMatrix(formula = ~ age + credit.score + email +
                    distance.to.store + online.visits + online.trans + 
                    online.spend + store.trans + store.spend, 
                  data=cust.df, diagonal="histogram")
## handle continuous variables only

library(gpairs)
gpairs(cust.df[, c(2:10]) ## both discrete and continuous variables


# 23. To perform conditional evaluation on every element of a vector
ifelse(x > 1, "hi", "bye")


# 24. Use aggregate function
aggregate(kids ~ Segment + ownHome, data=seg.df, sum)
aggregate(formula, data, FUN)


# 25. Plot histogram with formula notations
library(lattice)
histogram(~subscribe | segment, data = seg.df)


# 26. Plot boxplot
boxplot(income ~ Segment, data=seg.df, yaxt="n", ylab="Income ($k)")
ax.seq <- seq(from=0, to=120000, by=20000)
axis(side=2, at=ax.seq, labels=paste(ax.seq/1000, "k", sep=""), las=1)

# OR

library(lattice)
bwplot(Segment ~ income, data=seg.df, horizontal=TRUE, xlab = "Income")
# add conditioning variable
bwplot(Segment ~ income | ownHome, data=seg.df, horizontal=TRUE, 
       xlab="Income")

# 27. apply function to dataframe
mapply(df, df$Residual, df$Upper, df$Lower)


#28. replace NA's in a tibble
library(tidyr)
df <- tibble::tibble(x = c(1, 2, NA), y = c("a", NA, "b"), z = list(1:5, NULL, 10:20))
df %>% replace_na(list(x = 0, y = "unknown")) %>% str()


# 29.
