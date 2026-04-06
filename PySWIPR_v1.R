# Trying to find metrics we can use to filter out debris signal...

library(tidyverse)
library(data.table)
library(ggpubr)

input_data <- "worm_results.csv"
output_path <- "worm_results_filtered.csv"

raw_data <- as_tibble(fread(input_data))

# First step: Solidity filter...for s > 1, they are impossible datapoints for a real worm (low quality)
# because worms have actual area

solidity_filtered <- raw_data %>% 
  filter(Solidity <= 1)
fwrite(solidity_filtered, "worm_results_1solidity.csv")
# Great results - ts is carrying me lol

# Second step: Eliminate objects with short cumulative existing time
# by counting the number of frames they exist

et_df <- solidity_filtered %>% 
  group_by(TimeID, Well, ObjectID) %>% 
  summarise(totalframes = n())
b25 <- as.numeric(summary(et_df$totalframes)[2]) # Grabs the bottom 25% existence time value
et_df_filtered <- et_df %>% 
  filter(totalframes >= b25)

setDT(solidity_filtered)
setDT(et_df)
setDT(et_df_filtered)

x <- et_df_filtered[
  solidity_filtered,
  on = list(TimeID = TimeID, Well = Well, ObjectID = ObjectID),
  nomatch = NULL
]

solidity_et_filtered <- as_tibble(x)
fwrite(solidity_et_filtered, "worm_results_2solidityEt.csv")

# For objects overlapping each other, try resolving so that we only keep the object with the greater sd of angle
df1 <- solidity_et_filtered %>% 
  group_by(TimeID, Well, ObjectID) %>% 
  summarise(
    Frame_start = min(Rel_Frame),
    Frame_end = max(Rel_Frame),
    SD_angle = sd(Angle)
  )
df2 <- solidity_et_filtered %>% 
  group_by(TimeID, Well, ObjectID) %>% 
  summarise(
    Frame_start = min(Rel_Frame),
    Frame_end = max(Rel_Frame),
    SD_angle = sd(Angle)
  )
setDT(df1)
setDT(df2)
x <- df1[
  df2,
  on = list(TimeID = TimeID, Well = Well),
  nomatch = NULL,
  allow.cartesian = TRUE
]
overlapping <- as_tibble(x) %>% 
  filter(ObjectID != i.ObjectID & ((i.Frame_end < Frame_end & i.Frame_end > Frame_start) | (i.Frame_start < Frame_end & i.Frame_start > Frame_start)))
setDT(overlapping)
OL1 <- overlapping[SD_angle > i.SD_angle] # Gives all overlapping ObjectIDs that are probably not debris
OL2 <- overlapping[SD_angle < i.SD_angle] # Gives all overlapping i.ObjectIDs that are probably not debris

OL1_t <- as_tibble(OL1)
OL2_t <- as_tibble(OL2)

OL1_t_dobjs <- OL1_t %>%  # Gives i.ObjectIDs of all suspected debris
  select(TimeID, Well, i.ObjectID) %>% 
  rename(ObjectID = i.ObjectID) %>% 
  unique()
OL2_t_dobjs <- OL2_t %>% # Gives ObjectIDs of all suspected debris
  select(TimeID, Well, ObjectID) %>% 
  unique()

# Combine the dataframe of debris
cOL <- rbind.data.frame(OL1_t_dobjs, OL2_t_dobjs) %>% unique()

setDT(cOL)
setDT(solidity_et_filtered)

solidity_et_ol_filtered <- as_tibble(solidity_et_filtered[
  !cOL,
  on = list(TimeID = TimeID, Well = Well, ObjectID = ObjectID)
])
# Overlapping-removed ??? lets check 
fwrite(solidity_et_ol_filtered, "worm_results_3solidityEtOl.csv")
# Holy SHIT it worked. I have CLEAN DATA that i can finally start working on the counting logic...holyy shit...

# Summary: 3 step filtering process
# Step 1: Remove all datapoints (no matter the object) that have a solidity S > 1 (impossible for objects with real area).
# Step 2: Remove all objects that have datapoints less than the bottom 1st quartile number of points
# Step 3: For overlapping objects, remove the object that has a smaller standard deviation
# Note after step 3, all objects remaining in a single timeframe are assumed to be the same object, aka the worm

# Another filtering step: Remove group(Well, TimeID)s that do not have enough points to extrapolate
# thrashes from
# This df gives the number of points in a group(Well, TimeID)
totalpoints_df <- solidity_et_ol_filtered %>% 
  group_by(TimeID, Well) %>% 
  summarise(rows = n())
# Need a non-relative (e.g., not SD-based) threshold number of points to determine if the group(Well, TimeID)
# should be scrapped or not
# To be conservative and to increase quality of data, scrap the group(Well, TimeID) if
# there are less than 900 frames within a group(Well, TimeID), just omit the entire thing cuz its low quality
# 1800 * 0.5 = 900, so this is the same thing as "if missing 50% or more data, just omit".
totalpoints_filtered_df <- totalpoints_df %>% 
  filter(rows > 900) %>% 
  select(TimeID, Well)
setDT(totalpoints_filtered_df)
setDT(solidity_et_ol_filtered)
m <- solidity_et_ol_filtered[
  totalpoints_filtered_df,
  on = list(TimeID = TimeID, Well = Well),
  nomatch = NULL
]
solidity_et_ol_tp_filtered <- as_tibble(m)

fwrite(solidity_et_ol_tp_filtered, "worm_results_4solidityEtOlTp.csv")
# Then the data cleaning part is done, I will name this thing PySWIPR since we use R to clean PySWIP outputs