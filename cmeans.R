library(dplyr)
df <- read.csv("data/train1.csv", sep = ",", na.string = c("", " ", "NA"))
X <- data.frame(df$x_0, df$x_1)

df_train <- X |> filter(!is.na(df.x_1))
df_test  <- X |> filter(is.na(df.x_1))


library(e1071)
set.seed(42)
df_train <- scale(df_train)
cm <- cmeans(x = df_train, centers = 3, iter.max =  1000, dist = "euclidean", m=2)
head(cm$membership)

library(factoextra)
fviz_cluster(list(data = df_train, cluster = cm$cluster),
            ellipse.type = "norm",
            ellipse.level = 0.68,
            pallete = "jco",
            ggtheme = theme_classic())

df_test <- scale(df_test)


imputa_fcm <- function(dados_treino, dados_incompletos, k_clusters = 3, m_fuzzifier = 2, inter = 1000) {
  
  # A. Treina o FCM apenas na base limpa (sem NAs)
  modelo_fcm <- cmeans(dados_treino, centers = k_clusters, m = m_fuzzifier, iter.max = inter, dist = "euclidean")
  centros <- modelo_fcm$centers
  
  dados_imputados <- dados_incompletos
  
  # B. Itera por cada linha que possui valores ausentes
  for (i in 1:nrow(dados_incompletos)) {
    linha <- dados_incompletos[i, ]
    
    colunas_validas  <- which(!is.na(linha))
    colunas_ausentes <- which(is.na(linha))
    
    # Se não houver NA na linha, pula
    if (length(colunas_ausentes) == 0) next
    
    # Extrai o ponto e os centros considerando APENAS as colunas válidas
    ponto_conhecido <- as.numeric(linha[colunas_validas])
    centros_validos <- centros[, colunas_validas, drop = FALSE]
    
    # Distância euclidiana apenas nas dimensões conhecidas
    distancias <- sqrt(rowSums((t(t(centros_validos) - ponto_conhecido))^2))
    distancias[distancias == 0] <- 1e-10
    
    # Calcula a pertinência fuzzy (u_j) com base nas colunas conhecidas
    power <- 2 / (m_fuzzifier - 1)
    pertinencia <- numeric(k_clusters)
    for (j in 1:k_clusters) {
      pertinencia[j] <- 1 / sum((distancias[j] / distancias)^power)
    }
    
    # C. Imputa os NAs multiplicando a pertinência pelos centros das colunas ausentes
    for (col in colunas_ausentes) {
      dados_imputados[i, col] <- sum(pertinencia * centros[, col])
    }
  }
  
  return(dados_imputados)
}

imput = data.frame(imputa_fcm(df_train, df_test, 3, 2, 10000))
imput["type"] = "cmeans"

imput <- rename(imput, x_0 = df.x_0, x_1 = df.x_1)
head(imput)

df_original <- read.csv("data/original1.csv")
df_original["type"] = "original"

df_compare <- rbind(df_original, imput)
ggplot(df_compare, mapping = aes(x=x_0, y=x_1, fill = type))+
  geom_point(aes(color = type))


