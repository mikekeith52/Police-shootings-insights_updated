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
