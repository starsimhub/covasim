# Simplest possible Covasim usage example, in R.
#
# Run with e.g.:
#
#    Rscript simple.R

library(reticulate)
cv <- import('covasim')
sim <- cv$Sim()
sim$run()
sim$plot()