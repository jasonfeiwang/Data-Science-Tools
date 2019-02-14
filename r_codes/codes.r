# 0. Misc
## abort an expression: press ESC key
## show the results of a new assignment: wrap it with ()
##


# 1. convert ts object into data.frame
library(zoo)
df_flu = data.frame(month = as.Date(as.yearmon(time(flu))), value = as.matrix(flu))


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


# 7. ggplot graphics grammar
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


# 9. show a scatter plot of x, y and a smoothed line
tibble.name %>% 
  ggplot(mapping = aes(x = x, y = y)) +
    geom_point() + 
    geom_smooth(se = FALSE)


# 10. count the rows by a variable
diamonds %>% count(cut)
diamonds %>% group_by(cut) %>% summarise(n())


# 11. compare density of different variables (area under each histogram is equal to 1 for fair comparison)
ggplot(data = diamonds, mapping = aes(x = price, y = ..density..)) + 
  geom_freqpoly(mapping = aes(colour = cut), binwidth = 500)


# 12. create heatmap by count
diamonds %>% 
  count(color, cut) %>%  
  ggplot(mapping = aes(x = color, y = cut)) +
    geom_tile(mapping = aes(fill = n))


# 13.
