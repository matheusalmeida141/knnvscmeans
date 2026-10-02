library(cluster)
library(factoextra)
library(tidyverse)
data("USArrests")

head(USArrests)

df <- scale(USArrests)
head(df)


ggplot(df, mapping = aes(x = Murder, y = Rape))+
  geom_point()

res.fanny <- fanny(df, 3)
head(res.fanny$membership)

res.fanny$coeff

fviz_cluster(res.fanny,
            ellipse.type = "norm",
            repel = TRUE,
            pallete = "jco",
            ggtheme = theme_minimal(),
            legend = "right")

fviz_silhouette(res.fanny, pallete = "jco", ggtheme = theme_minimal())



library(e1071)
set.seed(42)
ss <- sample(1:50, 20)
df <- scale(USArrests[ss, ])
head(df)

cm <- cmeans(df, centers = 3, m = 2)
head(cm$membership)
head(cm$cluster)

library(corrplot)
corrplot(cm$membership, is.corr = F)

fviz_cluster(list(data = df, cluster = cm$cluster),
            ellipse.type = "norm",
            ellipse.level = 0.68,
            pallete = "jco",
            ggtheme = theme_classic())
