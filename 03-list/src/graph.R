library(tidyverse)
library(ggplot2)

mh <- function(N, d_vc) N**d_vc

original <- function(N, delta, d_vc) {
  sqrt((8 / N) * log(4 * mh(2 * N, d_vc) / delta))
}

rademacher <- function(N, delta, d_vc) {
  sqrt(2 * log(2 * N * mh(N, d_vc)) / N) + sqrt(2 * log(1 / delta) / N) + 1 / N
}

parrondo <- function(N, delta, d_vc) {
  1 / N + 0.5 * sqrt(4 / N**2 + 4 / N * log(6 * mh(2 * N, d_vc) / delta))
}

devroye <- function(N, delta, d_vc) {
  (2 / N + sqrt(4 / N**2 + 4 * (1 - 2 / N) * (1 / (2 * N)) * (log(4) + 2 * d_vc * log(N) - log(delta)))) / (2 * (1 - 2 / N))
}

N <- seq(5, 10000)
delta <- 0.05
d_vc <- 50

data <- data.frame(
  x = N,
  original = original(N, delta, d_vc),
  rademacher = rademacher(N, delta, d_vc),
  parrondo = parrondo(N, delta, d_vc),
  devroye = devroye(N, delta, d_vc)
)

ggplot(data, aes(x = N)) +
  geom_line(aes(y = original, colour = "Original")) +
  geom_line(aes(y = rademacher, colour = "Rademacher")) +
  geom_line(aes(y = parrondo, colour = "Parrondo")) +
  geom_line(aes(y = devroye, colour = "Devroye")) +
  xlab("N (samples)") +
  ylab("generalization error (bound)") +
  ylim(c(0, 5))
