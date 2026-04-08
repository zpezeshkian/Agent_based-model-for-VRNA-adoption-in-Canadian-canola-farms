
######Calculating gross margin using Bio-economic model######
###Zahra Pezeshkian Nov. 2025##
rm(list = ls())
library(lpSolve)
library (dplyr)
library(tidyr)
library(ggplot2)
library(readr)
set.seed(123)
options(scipen = 999)
####1. Load csv dataset for farm size and number of agents
setwd("/Users/zahrapezeshkian/Desktop/Simpson/Simulation/ABM/Data/Others")
farm_data_census <- read.csv("New_data.csv")
farm_data_census <- farm_data_census[!is.na(farm_data_census$Farm_Size_ha),]
###Number of agents###
num_agents <- nrow(farm_data_census)
farmers_ids <- farm_data_census$Farmer_ID ####For tracking
farm_sizes_ha <- farm_data_census$Farm_Size_ha

# Confirm data loading
cat("Number of agents loaded from CSV:", num_agents, "\n")
cat("First few farm sizes (ha):", head(farm_sizes_ha), "\n")

##2. General parameters and constants##
num_landscapes <- 12
num_price_draws <- 5
PN <- 1.35 ###price of nitrogen $/kg
N_max_practical <- 189 #### kg/ha - Practical upper bound for N application (based on Table 3 highest optimal N)

####cost parameters#####
Coperating <- 499 ###CAD per ha (general operating cost excluding N cost)

total_VRT_cost <- 260612 ###CAD per ha (including equipment costs)
share_VRNA <- 0.20
VRNA_total_cost <- total_VRT_cost * share_VRNA

CVRNA <- VRNA_total_cost / farm_data_census$Farm_Size_ha  # $/ha

#### 3.Yield response functionc (table 3)#####
##Function for yield (Y) = b0 + b1*N - b2*N^2
yield_response_params <- data.frame(
  Zone = c("Low", "Medium", "High"),
  b0 = c(2457, 3026, 3230),
  b1 = c(18.1, 8.72, 5.30),
  b2 = c(0.13, 0.04, 0.014),
  Optimal_N_doc = c(69, 109, 189) 
)

### 4.Canola price draws#####
Canola_prices_per_tonne <- runif (num_price_draws, min= 550, max= 685) ##five random price draws from a uniform distr.
Pcanola_per_kg <- Canola_prices_per_tonne / 1000 # Convert to CAD/kg

### 5.Landscape templates generation#####
Create_synthetic_landscape <- function(id, composition, config){
  num_cells_x <- 20
  num_cells_y <- 20
  total_cells <- num_cells_x*num_cells_y
  
  composition <- composition/sum(composition) ##sum to 1
  
  ##Assign zones based on compositions
  num_low <- round(total_cells*composition[1])
  num_med <- round(total_cells*composition[2])
  num_high <- total_cells - num_low - num_med
  
  zones_flat <- c(rep(1, num_low), rep(2, num_med), rep(3, num_high)) #1=Low, 2=Medium, 3= High
  
  landscape_matrix <- matrix(0, nrow=num_cells_x, ncol=num_cells_y)
  
  if(config == "random"){
    landscape_matrix[] <- sample(zones_flat)
  }
  else if(config == "clustered_high"){
    block_size <- round(sqrt(num_high))
    if (block_size > num_cells_x) block_size <- num_cells_x
    start_x <- num_cells_x - block_size + 1
    start_y <- num_cells_y - block_size + 1
    landscape_matrix[start_x:num_cells_x, start_y:num_cells_y] <- 3 #High
    remaining_indices <- which(landscape_matrix == 0)
    landscape_matrix[remaining_indices] <- sample(c(rep(1, num_low), rep(2, num_med)),
                                                  size = length(remaining_indices), replace = TRUE)
  }else if (config == "clustered_low"){
    block_size <- round(sqrt(num_low))
    if (block_size > num_cells_x) block_size <- num_cells_x
    landscape_matrix[1:block_size, 1:block_size] <- 1 # Low
    remaining_indices <- which(landscape_matrix == 0)
    landscape_matrix[remaining_indices] <- sample(c(rep(2, num_med), rep(3, num_high)),
                                                  size = length(remaining_indices), replace = TRUE)
  }else if (config == "checkerboard"){
    temp_zones_to_assign <- c(rep(1, num_low), rep(2, num_med), rep(3, num_high))
    landscape_matrix[] <- sample(temp_zones_to_assign, size = total_cells, replace = FALSE)
  }
  
  return(landscape_matrix)
}
####Define compositions and configurations from table 4 for 12 L####

###shares of compositions
compositions <- list(
  c(0.15, 0.25, 0.60), c(0.15, 0.25, 0.60), c(0.15, 0.25, 0.60), c(0.15, 0.25, 0.60),
  c(0.33, 0.33, 0.34), c(0.33, 0.33, 0.34), c(0.33, 0.33, 0.34), c(0.33, 0.33, 0.34),
  c(0.60, 0.25, 0.15), c(0.60, 0.25, 0.15), c(0.60, 0.25, 0.15), c(0.60, 0.25, 0.15)
)

configurations <- c(
  "clustered_high", "clustered_low", "random", "checkerboard",
  "clustered_high", "clustered_low", "random", "checkerboard",
  "clustered_high", "clustered_low", "random", "checkerboard"
)

num_landscapes <- length(compositions)

landscape_templates <- list()
for(i in 1:num_landscapes){
  landscape_templates[[i]] <- Create_synthetic_landscape (
    id = paste0("L", i),
    composition = compositions[[i]],
    config = configurations[[i]] 
  )
}

### 6. bio-economic optimization####
##Function to calculate profit for a given N rate in a specific zone
##Yj = b0 + b1*Nj - b2*Nj^2
# Profit_j = (Pcanola * Yj) - (PN * Nj) - Coperating
calculate_zone_profit <- function(N_rate, Pcanola_per_kg, PN_price, zone_params, Coperating_cost){
  N_rate <- max(0, N_rate)
  ##yield response for a specific zone
  b0 <- as.numeric(zone_params$b0)
  b1 <- as.numeric(zone_params$b1)
  b2 <- as.numeric(zone_params$b2)
  
  #calculate yield
  yield_kg_ha <- b0 + b1*N_rate - b2*(N_rate^2)
  yield_kg_ha <- max(0, yield_kg_ha) ###non_negative yield
  
  gross_margin_zone <- (Pcanola_per_kg*yield_kg_ha) - (PN_price*N_rate) - Coperating_cost
  
  return(gross_margin_zone)
}


##Function to find optimal N rate for a specific zone###
##Nj* = (b1- (PN/PCanola)) / (2*b2)
find_optimal_N_for_zone_analytical <- function(Pcanola_per_kg, PN_price, zone_params){
  
  ##Get yield response parameters for specific zone
  b1 <- as.numeric(zone_params$b1)
  b2 <- as.numeric(zone_params$b2)
  
  #Calculate optimal N rate using FOC
  if (Pcanola_per_kg <= 0 || b2 <= 0){
    optimal_N <- 0
  }else {
    optimal_N <- (b1 - (PN_price/Pcanola_per_kg))/ (2*b2)
  }
  
  optimal_N <- max(0, optimal_N) ##Non-negative N
  optimal_N <- min(optimal_N, N_max_practical)
 
  return(optimal_N) 
}

###Average optimal N for a landscape under VRNA

calculate_vrna_avgN_per_ha <- function(landscape_matrix, Pcanola_per_kg, PN_price,
                                       yield_response_params){
  total_cells <- length(landscape_matrix)
  zone_counts <- table(factor(landscape_matrix, levels = c(1,2,3)))
  zone_shares <- as.numeric(zone_counts/total_cells)
  
  zone_optN <- numeric(3)
  for (i in 1:3){
    zone_name <- c("Low", "Medium", "High")[i]
    current_zone_params <- yield_response_params[yield_response_params$Zone == zone_name,]
    zone_optN[i] <- find_optimal_N_for_zone_analytical(Pcanola_per_kg, PN_price, current_zone_params)
  }
  
  ###Area_weighted average N/ha across zones    
  
  avgN_vrna_ha <- sum(zone_optN * zone_shares)
  return(avgN_vrna_ha)
}


###Function to calculate profit for 1-hectare landscape under VRNA
calculate_vrna_total_profit <- function(landscape_matrix, Pcanola_per_kg, PN_price,
                                        yield_response_params, Coperating_cost, CVRNA_cost){
  
  total_cells<- length(landscape_matrix)
  zone_counts <- table(factor(landscape_matrix, levels = c(1,2,3)))
  zone_shares <- zone_counts/total_cells 
  
  zone_profits_per_ha <- numeric(3) #numeric vector of length 3 
                                    #initialize profits for each zone type
  
  for(i in 1:3){
    zone_name <- c("Low","Medium","High")[i]
    current_zone_params <- yield_response_params[yield_response_params$Zone == zone_name,]
    optimal_N_zone <- find_optimal_N_for_zone_analytical(Pcanola_per_kg, PN_price,
            current_zone_params) #finding optimal N for a specific zone
    zone_profits_per_ha[i] <- calculate_zone_profit(
      N_rate = optimal_N_zone,
      Pcanola_per_kg = Pcanola_per_kg,
      PN_price = PN_price,
      zone_params = current_zone_params,
      Coperating_cost = Coperating_cost
    ) #calculating profit for this zone at the optimal N rate
    
  }
  
  
  #Calculate the average profit per hectare for this 1-ha landscape under VRNA
  profit_per_ha_landscape_before_vrna_cost <-sum(zone_profits_per_ha*zone_shares) 
  total_vrna_profit_per_ha <- profit_per_ha_landscape_before_vrna_cost - CVRNA_cost #Subtract VRNA specific costs (per hectare)
  
  return(total_vrna_profit_per_ha)
}

##Function to calculate total URNA profit for a given 1-hectare landscape
calculate_urna_total_profit <- function(landscape_matrix, Pcanola_per_kg, PN_price, yield_response_params,
                                        Coperating_cost, N_urna_rate = 108) { # N_urna_rate from document's average
  total_cells <- length(landscape_matrix)
  zone_counts <- table(factor(landscape_matrix, levels = c(1, 2, 3)))
  zone_shares <- zone_counts / total_cells
  
  zone_urna_profits_per_ha <- numeric(3)
  
  for (i in 1:3) {
    zone_name <- c("Low", "Medium", "High")[i]
    current_zone_params <- yield_response_params[yield_response_params$Zone == zone_name, ]
    # Calculate profit for this zone using the *uniform* N rate
    zone_urna_profits_per_ha[i] <- calculate_zone_profit(N_urna_rate, Pcanola_per_kg, PN_price, current_zone_params, Coperating_cost)
  }
  
  # Calculate the average profit per hectare for this 1-ha landscape under URNA
  total_urna_profit_per_ha <- sum(zone_urna_profits_per_ha * zone_shares)
  
  return(total_urna_profit_per_ha)
}


### 7. Simulations for all agents, over all landscapes and five price draws###
results_df <- data.frame(
  Agent_ID = integer(),
  Farm_Size_ha = numeric(),
  Landscape_ID = integer(),
  Price_Draw_ID = integer(),
  Pcanola_per_kg = numeric(),
  Gross_Margin_VRNA_per_ha = numeric(),
  Gross_Margin_URNA_per_ha = numeric(),
  Gross_Margin_VRNA_Total_Farm = numeric(),
  Gross_Margin_URNA_Total_Farm = numeric(),
  N_URNA_kg_ha = numeric(),
  N_VRNA_kg_ha = numeric(),
  N_Reduction_kg_ha = numeric(),
  N_Reduction_total_farm_kg = numeric()
)
  
 #Loop through each agent
for(a_idx in 1:num_agents){
  agent_id_current <- farmers_ids[a_idx]
  current_farm_size <- farm_sizes_ha[a_idx]
  
  #Loop through each landscape template
  for(l in 1:num_landscapes){
    landscape_id_current <- l
    current_landscape <- landscape_templates[[l]]
    
    #Loop through each canola price draw
    for(p in 1:num_price_draws){
      price_draw_id_current <- p
      current_Pcanola <- Pcanola_per_kg[p]
      
      # Calculate VRNA gross margin per hectare for the current landscape and price
      gm_vrna_per_ha <- calculate_vrna_total_profit(
        landscape_matrix = current_landscape,
        Pcanola_per_kg = current_Pcanola,
        PN_price = PN,
        yield_response_params = yield_response_params,
        Coperating_cost = Coperating,
        CVRNA_cost = CVRNA[a_idx]
      )
      
      # Calculate URNA gross margin per hectare for the current landscape and price
      gm_urna_per_ha <- calculate_urna_total_profit(
        landscape_matrix = current_landscape,
        Pcanola_per_kg = current_Pcanola,
        PN_price = PN,
        yield_response_params = yield_response_params,
        Coperating_cost = Coperating
      )
      
      #compute VRNA avg optimal N (kg/ha) for this landscape and price
      N_vrna_ha <- calculate_vrna_avgN_per_ha(
        landscape_matrix = current_landscape,
        Pcanola_per_kg = current_Pcanola,
        PN_price = PN,
        yield_response_params = yield_response_params
      )
      # Baseline (URNA) uniform N rate (kg/ha)
      N_urna_ha <- 108
      
      # N reduction (kg/ha) - cap at 0 to avoid negative reduction
      N_red_ha <- max(0, N_urna_ha - N_vrna_ha)
      
      # Total farm reduction (kg)
      N_red_total_farm <- N_red_ha * current_farm_size
      
      # Calculate total gross margin for the farm (per-ha * farm_size)
      gm_vrna_total_farm <- gm_vrna_per_ha * current_farm_size
      gm_urna_total_farm <- gm_urna_per_ha * current_farm_size
      
      # Add results to the data frame
      results_df <- rbind(results_df, data.frame(
        Agent_ID = agent_id_current,
        Farm_Size_ha = current_farm_size,
        Landscape_ID = landscape_id_current,
        Price_Draw_ID = price_draw_id_current,
        Pcanola_per_kg = current_Pcanola,
        
        Gross_Margin_VRNA_per_ha = gm_vrna_per_ha,
        Gross_Margin_URNA_per_ha = gm_urna_per_ha,
        Gross_Margin_VRNA_Total_Farm = gm_vrna_total_farm,
        Gross_Margin_URNA_Total_Farm = gm_urna_total_farm,
        
        N_URNA_kg_ha = N_urna_ha,
        N_VRNA_kg_ha = N_vrna_ha,
        N_Reduction_kg_ha = N_red_ha,
        N_Reduction_total_farm_kg = N_red_total_farm
      ))
      
    }
    
  }
}

results_df <- results_df %>%
  mutate(
    Benefit_better_N_targeting_total_farm =
      (Gross_Margin_VRNA_Total_Farm) - Gross_Margin_URNA_Total_Farm
  )


#############Save the results###############
write.csv(results_df, "GM_results.csv", row.names = FALSE)
getwd()

#############Creating reference income using gross margins of URNA##########

setwd("/Users/zahrapezeshkian/Desktop/Simpson/Simulation/ABM/Data/Others")
gm <-read_csv("GM_results.csv")

urna_summary <- gm %>%
  group_by(Agent_ID) %>%
  summarise(
    GM_min = min(Gross_Margin_URNA_Total_Farm, na.rm = TRUE),
    GM_max = max(Gross_Margin_URNA_Total_Farm, na.rm = TRUE),
    GM_mean = mean(Gross_Margin_URNA_Total_Farm, na.rm = TRUE)
  )

print(urna_summary)

farm_data_census <- farm_data_census %>%
  left_join(urna_summary, by = c("Farmer_ID" = "Agent_ID")) %>%
  mutate(
   Vref = GM_min + (farm_income_importance -1)/4 * (GM_max - GM_min)
  )
write.csv(farm_data_census, "New_data.csv")


#########Summary of Statistics#################
neg_agent_total <- unique(results_df$Agent_ID[results_df$Gross_Margin_VRNA_Total_Farm < 0])
neg_agent_total

Positive_agnet_total <- unique(results_df$Agent_ID[results_df$Gross_Margin_VRNA_Total_Farm>0])
Positive_agnet_total

neg_rows <- subset(results_df,Gross_Margin_VRNA_Total_Farm < 0)
unique_neg_farm_size <- sort(unique(neg_rows$Farm_Size_ha))
unique_neg_farm_size


neg_agent_info <- results_df %>%
  filter(Gross_Margin_VRNA_Total_Farm <0) %>%
  distinct(Agent_ID, Farm_Size_ha) %>%
  arrange(Farm_Size_ha)

neg_agent_info

positive_agent_info <- results_df %>%
  filter(Gross_Margin_VRNA_Total_Farm >0) %>%
  distinct(Agent_ID, Farm_Size_ha) %>%
  arrange(Farm_Size_ha)
positive_agent_info

summary(positive_agent_info$Farm_Size_ha)

always_neg <- results_df %>%
  group_by(Agent_ID, Farm_Size_ha) %>%
  summarise(
    n_obs = n(),
    n_neg = sum(Gross_Margin_VRNA_Total_Farm < 0),
    .groups = "drop"
  ) %>%
  filter(n_neg == n_obs)

always_neg

agent_summary <- results_df %>%
  group_by(Agent_ID, Farm_Size_ha) %>%
  summarise(
    n_obs = n(),
    n_neg = sum(Gross_Margin_VRNA_Total_Farm <0),
    .groups = "drop"
  )
n_always_neg <- sum(agent_summary$n_neg == agent_summary$n_obs, na.rm = TRUE)
n_always_neg

always_neg <- agent_summary %>%
  filter(!is.na(n_neg), !is.na(n_obs)) %>%
  filter (n_neg == n_obs)
always_neg
sum(agents_input_with_CE$CE_CPT_delta_profit<0, na.rm = TRUE)

neg_ce <- agents_input_with_CE %>%
  filter(!is.na(CE_CPT_delta_profit), CE_CPT_delta_profit<0) %>%
  select(Farmer_ID, Farm_Size_ha, CE_CPT_delta_profit)%>%
  arrange(Farm_Size_ha)

print(neg_ce)

pos_ce <- agents_input_with_CE %>%
  filter(!is.na(CE_CPT_delta_profit), CE_CPT_delta_profit>0) %>%
  select(Farmer_ID, Farm_Size_ha, CE_CPT_delta_profit)%>%
  arrange(Farm_Size_ha)

print(pos_ce)


