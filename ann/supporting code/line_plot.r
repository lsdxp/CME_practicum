library(plotly)
setwd("/Users/Heran/Desktop/temp/Apr\ 15th/time\ series")
data <- read.csv("plot.csv", header = TRUE)
time <- data$t
toString(time)
price <- data$tp2
vol <- data$vol
acc = data$acc
interval <- c(13790:14350)
subtime <- time[interval]
subprice <- price[interval]
subacc <- acc[interval]
subvol <- vol[interval]
a <- list(autotick = FALSE, title = "Time", ticks = "outside", dtick = 120,  tickwidth = 0, ticklen = 0)
ay <- list(tickfont = list(color = "red"), title = "Accuracy", overlaying = "y", side = "right")
plot_ly(x=subtime, y= subprice, name = "Trade Price")%>%
  add_trace(x=subtime, y = subacc, name = "Volatility", yaxis = "y2")%>%
  add_trace(x=subtime, y = subvol, name = "Volatility", yaxis = "y2", colors = "r")%>%
  layout(xaxis = a, yaxis2 = ay)

plot_ly(x=subtime, y= subprice, name = "Trade Price")%>%
  add_trace(x=subtime, y = subacc, name = "Accuracy", yaxis = "y2")
# plot_ly(x=acc, y=vol, name = "Accuracy v.s. Volatility", mode = "markers", opacity = vol)

a <- list(autotick = FALSE, ticks = "outside", dtick = 1, tickwidth = 0, ticklen = 0, tickcolor = toRGB("blue"))
s <- seq(1, 4, by = 0.25)
plot_ly(x = s, y = s) %>%
  layout(xaxis = a, yaxis = a)
