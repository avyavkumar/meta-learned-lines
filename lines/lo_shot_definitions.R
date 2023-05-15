library(MASS)

xwx <- function(xtx_in, x, w) {

  # computes inverse of Z'Z,
  # where Z = rows of X with non-zero weight

  # xtx_in = inverse of X'X
  # x = n x p covariate matrix X
  # w = length n vector of zero-one weight
  # w cannot be all-zero
  if (sum(w) == 0) stop('All rows have zero weight.')



  # case 1: no zero-weights; same as inverse of X'X
  if (prod(w) == 1) return(xtx_in)

  # case 2: number of zero-weight rows >= p
  # invert directly
  else if (sum(w) < ncol(x)) {
    z1 <- x[w == 1, , drop = F]
    return(ginv(t(z1) %*% z1))
  }

  # case 3: number of zero-weight rows < p
  # use woodbury identity to invert a smaller matrix
  else {
    z0 <- x[w == 0, , drop = F]
    A <- z0 %*% xtx_in
    B <- -A %*% t(z0)
    diag(B) <- diag(B) + 1
    return(xtx_in + t(A) %*% ginv(B) %*% A)
  }
}

xwy <- function(x, y, w) {

  # computes Z'y,
  # where Z are the rows of x with non-zero weight

  # x = n x p covariate matrix X
  # y = length n vector
  # w = length n vector of zero-one weight
  # w cannot be all-zero
  if (sum(w) == 0) stop('All rows have zero weight.')

  # set zero-weight entries of y to be zero
  y[w == 0] <- 0
  return(t(x) %*% y)
}

#library(MASS)
#x <- matrix(runif(5 * 1), 5, 1)
#xtx_in <- solve(t(x) %*% x)
#w <- c(1,0,0,0,0)
#y <- runif(5)

#xwx(xtx_in, x, w)
#xwy(x, y, w)
#xwx(xtx_in, x, w) %*% xwy(x, y, w)

beta_w <- function(xtx_in, x, y, w) {
  xwx(xtx_in, x, w) %*% xwy(x, y, w)
}

two_norm <- function(a, b) {
  sqrt(sum((a - b)^2))
}


group_classes <- function(data, label, k) {
  mu <- t(sapply(unique(label), function(ii) {
    colMeans(data[label == ii, , drop = F])
    }))

  mu_dist <- dist(mu)
  cluster <- cutree(hclust(mu_dist, method = "complete"), k = k)

  mu2 <- t(sapply(unique(cluster), function(ii) {
    colMeans(mu[cluster == ii, , drop = F])
  }))

  dist2 <- as.matrix(dist(mu2))

  jj <- 1
  while (jj <= length(unique(cluster))) {
    #print(length(unique(cluster)))
    #print(jj)
    if (table(cluster)[jj] == 1) {
      new_cluster <- which(rank(dist2[jj, ]) == 2)
      cluster[cluster == jj] <- new_cluster
    }
    jj <- jj + 1
  }
  # print(cluster)
  return(cluster)
}

furthest_classes <- function(data, label, classes) {
  mu <- t(sapply(classes, function(ii) {
    colMeans(data[label == ii,  , drop = F])
  }))
  mu_dist <- as.matrix(dist(mu))
  furthest <- which(mu_dist == max(mu_dist), arr.ind = T)[1, ]
  return(classes[furthest])
}

add_classes <- function(data, label, classes, max_diff = 1.5) {

  # two furthest-apart classes in the group
  # we initially fit a line that pierces through both of their centroids
  furthest <- furthest_classes(data, label, classes)
  rest <- classes[!(classes %in% furthest)]

  x <- cbind(1, data[, -ncol(data)])
  y <- data[, ncol(data)]

  xtx_in <- ginv(t(x) %*% x)
  w <- ifelse(label %in% furthest, 1, 0)
  beta <- beta_w(xtx_in, x, y, w)

  while (length(rest) > 0) {

    # for the remaining classes, fit a regression line with it and
    # only classes currently in 'furthest' list
    beta_list <- lapply(rest, function(ii) {
      w <- ifelse(label %in% c(ii, furthest), 1, 0)
      return(beta_w(xtx_in, x, y, w))
    })

    # compare the distance between the regression line with the initial two furthest classes
    # and the newly fitted regression line
    distance <- sapply(beta_list, function(a) two_norm(a, beta))

    # stop if the smallest difference between the two regression lines is
    # greater than the max tolerance
    if (all(distance > max_diff)) {
      rest <- integer(0)
    } else {

      # otherwise, include the class whose addition resulted in the smallest change
      # in the original regression line
      add <- which.min(distance)[1]
      furthest <- c(furthest, rest[add])
      rest <- rest[-add]
    }
  }

  return(list(group = unique(furthest), line = beta))
}


order_classes <- function(data, label, group) {
  # first two elements in group must be the furthest away.
  # this will be the case if group comes from recursive regression

  if (length(group) == 1) {
    return(group)
  } else {
    temp <- sapply(group[-1], function(ii) {
      a <- colMeans(data[label == group[1], , drop = F])
      b <- colMeans(data[label == ii, , drop = F])
      return(sum((a - b)^2))
    })
    return(c(group[1], group[-1][order(temp)]))
  }
}



recursive_reg <- function(data, label, k, max_diff = 1e-1, keep_all = T) {

  label <- as.vector(label)

  # group the class-wise centroids into k groups
  init_group <- group_classes(data, label, k)
  k_new <- length(unique(init_group))

  #if (k_new == 1) {
  #  val <- list(group = order_classes(data, label, 1),
  #              line = lm())
  #}
  # for each of the k groups, find a line that incorporates
  # as many of the classes in that group as possible
  val <- lapply(sort(unique(init_group)), function(ii) {
    classes <- which(init_group == ii)
    # print(classes)
    if (keep_all) {
      temp <- add_classes(data, label, classes, max_diff)
      temp$group <- order_classes(data, label, temp$group)
      return(temp$group)
    } else {
      if (length(unique(classes)) == 1) {
        return(NULL)
      } else {
        temp <- add_classes(data, label, classes, max_diff)
        temp$group <- order_classes(data, label, temp$group)
        return(temp$group)
      }
    }
    #add_classes(data, label, classes, max_diff)
    #if (length(unique(classes)) == 1) {
    #  return(NULL)
    #} else {
    #  add_classes(data, label, classes, max_diff)
    #}
  })

  if (keep_all) {
    # if keep_all = T, keep lines from  single classes
    return(val)
  } else {
    # If keep_all = F, filter out groups with only one class
    return(val[lengths(val) != 0])
  }
}