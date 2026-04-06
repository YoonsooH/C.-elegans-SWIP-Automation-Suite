# Counts thrashes for observed time then adjusts counted thrashes to a thrash rate by normalizing time to 60 seconds
# i.e., adjusting thrashes to thrash rate

library(tidyverse)
library(data.table)
library(ggpubr)
library(pracma)

data_path <- "worm_results_4solidityEtOlTp.csv"
data <- as_tibble(fread(data_path))

# The inflection point of the angle vs frame plot is the number of 0-crossings in the second derivative plot
# Which is equal to the number of peaks and troughs in the first derivative plot
# We only want certain large inflection points; since small ones are noise from paralyzed worms

# Construct a new dataframe that has the columns
# TimeID, Well, Frames (number of datapoints), Thrashes (number of thrashes counted), Thrash rate (normalized to 1 min)

thrash_df_1 <- data %>% 
  group_by(TimeID, Well) %>% 
  summarise(Frames = n())
# NEW ONE; changed to velocity instead of using acceleration, since we are doing peak counting not
# 0-crossing of acceleration plot
# Biologically speaking, the minpeakheight represents the speed at which the worm is turning when it
# is hitting an inflection point; to a human observer, a worm that turns really slowly will
# probably not count it as a thrash, so this takes into account that too (sort of).
thrash_df_2 <- data %>% 
  select(TimeID, Well, Rel_Frame, Angle) %>% 
  group_by(TimeID, Well) %>% 
  mutate(
    Ang_vel = (Angle - lag(Angle)) / (Rel_Frame - lag(Rel_Frame)),
    Ang_acc = (Ang_vel - lag(Ang_vel)) / (Rel_Frame - lag(Rel_Frame)),
    lagAng_acc = lag(Ang_acc),
    signAcc = sign(Ang_acc),
    signLagAcc = lag(signAcc),
    diffAmp = abs(lagAng_acc - Ang_acc)
  ) %>% 
  na.omit() %>% 
  summarise(Thrashes = (length(findpeaks(Ang_vel, minpeakheight = 20, minpeakdistance = 8) + length(findpeaks(-Ang_vel, minpeakheight = 20, minpeakdistance = 8))) %/% 2))

setDT(thrash_df_1)
setDT(thrash_df_2)

k <- thrash_df_2[
  thrash_df_1,
  on = list(TimeID == TimeID, Well == Well),
  nomatch = NA
]

thrash_df_final <- as_tibble(k) %>% 
  mutate(`Thrash rate` = Thrashes / Frames * 1800) # Normalizing thrash count to thrash rate
thrash_df_final[is.na(thrash_df_final)] <- 0
fwrite(thrash_df_final, file = "thrash.csv")