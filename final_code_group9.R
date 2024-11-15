#load libraries
library(tidyverse)
library(dplyr)
library(tm)
library(stringr)
library(caret)
library(glmnet)
library(quanteda)
library(tokenizers)
library(ranger)
library(xgboost)
library(class)
library(lightgbm)
library(ggplot2)
library(maps)
library(wordcloud)
library(RColorBrewer)


#load data files
setwd("C:/Users/Yashvi Mohta/OneDrive/Desktop/Spring 2024/BUDT758T-Data Mining and Predictive Analytics/PROJECT")
train_x <- read_csv("airbnb_train_x_2024.csv")
train_y <- read_csv("airbnb_train_y_2024.csv")
test_x <- read_csv("airbnb_test_x_2024.csv")
train_extra<-read_csv("iou_zipcodes_2020.csv")

#Join the training y to the training x file
#Turn the target variables into factors
train <- cbind(train_x, train_y) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score),
         high_booking_rate = as.factor(high_booking_rate)) 
train_clean <- train %>%
  select(-high_booking_rate)
##EDA ON A FEW VARIABLES##
summary(train_clean$amenities)


#####----------------------DATA CLEANING AND PREPARATION--------------####

# Function to tokenize, normalize, and create binary indicators for a column
process_column <- function(data, column_name) {
  # Tokenize the column by splitting on commas
  data <- data %>%
    mutate(
      !!sym(column_name) := str_split(!!sym(column_name), ",")
    )
  # Normalize the column by converting to lowercase and removing special characters
  data <- data %>%
    mutate(
      !!sym(column_name) := lapply(!!sym(column_name), function(x) tolower(gsub("[[:punct:]]", "", x)))
    )
  # Remove leading and trailing whitespace
  data <- data %>%
    mutate(
      !!sym(column_name) := lapply(!!sym(column_name), function(x) trimws(x))
    )
  # Extract unique values from the column
  unique_values <- unique(unlist(data[[column_name]]))
  
  # Create binary indicators for each unique value
  binary_indicators <- sapply(unique_values, function(value) {
    sapply(data[[column_name]], function(value_list) {
      as.numeric(value %in% value_list)
    })
  })
  # Convert the binary indicators to a data frame
  binary_df <- as.data.frame(binary_indicators)
  
  # Prefix column names with 'has_'
  colnames(binary_df) <- paste0("has_", unique_values)

  # Bind the binary indicators to the original data frame
  data <- cbind(data, binary_df)
  
  # Remove the original column
  data <- data %>%
    select(-!!sym(column_name))
  
  return(data)
}


#--------------------------------------------------------------------------##
#EDA ON FEW VARIABLES
#host_neighbourhood
# Count NA values in host_neighbourhood
na_count <- sum(is.na(train_clean$host_neighbourhood))
non_na_count <- sum(!is.na(train_clean$host_neighbourhood))

# Plot NA and non-NA count
ggplot() +
  geom_bar(aes(x = c("NA", "Non-NA"), y = c(na_count, non_na_count)), stat = "identity", fill = c("skyblue", "salmon")) +
  labs(title = "Number of NA and Non-NA in host_neighbourhood", x = "", y = "Count")

# License
summary(train_clean$license)
# Count the frequency of each license value
license_counts <- table(train_clean$license)

# Convert license counts to data frame
license_df <- as.data.frame(license_counts)
names(license_df) <- c("License", "Count")

#interaction variable
summary(train_clean$interaction)


#------------------------------------------------------------------------#
#Variable Cleaning:-experiences_offered,smart_location,host_neighbourhood,bathrooms,
#monthly_price, license_status,host_verifications, amenities
airbnb_train <- train_clean %>%
  mutate(
    experiences_offered = as.factor(experiences_offered),
    smart_location = as.factor(smart_location),
    host_neighbourhood =ifelse(is.na(host_neighbourhood),"Not Mentioned", host_neighbourhood),
    host_neighbourhood = as.factor(host_neighbourhood),
    bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
    monthly_price = ifelse(is.na(monthly_price),mean(monthly_price, na.rm=TRUE), monthly_price),
    license_status = case_when(
      is.na(license) ~ "No",
      grepl("pending", license, ignore.case = TRUE) ~ "Pending",
      grepl("^\\d+$", license) ~ "Yes",
      TRUE ~ "Yes"
    ),
    license_status = as.factor(license_status)
  ) %>%
  mutate(
    city = trimws(tolower(city)),
    city = gsub("[[:punct:]]", "", city),
    city = gsub("\\s+", " ", city),
    city = ifelse(is.na(city), "Missing", city), # Handling missing values
    city = as.factor(city) 
  ) %>%
  select(-license)

airbnb_train <- airbnb_train %>%
  process_column("host_verifications") %>%
  process_column("amenities")
#___________Checking EDA---------------#
#host_neighbourhood
# Logical vectors 
simple_category <- (airbnb_train$host_neighbourhood == "Not Mentioned")
other_category <- (airbnb_train$host_neighbourhood != "Not Mentioned")

# Count each category
category_counts <- c(Not_Mentioned = sum(simple_category), Other_Categories = sum(other_category))

# Convert counts to data frame for ggplot
plot_data <- data.frame(Category = names(category_counts), Count = as.numeric(category_counts))

# Plotting
ggplot(plot_data, aes(x = Category, y = Count, fill = Category)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Not_Mentioned" = "skyblue", "Other_Categories" = "salmon")) +
  labs(title = "Number of 'Not Mentioned' and 'Other Categories' in host_neighbourhood",
       x = "", y = "Count") +
  theme_minimal()
##License_Status
# Create data frame for plotting
license_data <- data.frame(
  License_Status = c("No", "Pending", "Yes"),
  Count = c(83919, 4759, 3389)
)
# Plotting
ggplot(license_data, aes(x = License_Status, y = Count, fill = License_Status)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("No" = "skyblue", "Pending" = "salmon", "Yes" = "green")) +
  labs(title = "Distribution of License Status",
       x = "License Status",
       y = "Count") +
  theme_minimal()

#HOST
#Variable Cleaning:-accommodates,availability_90,guests_included,host_total_listings_count,
#latitude, security_deposit,minimum_nights, cancellation_policy
airbnb_train <- airbnb_train %>% 
  ungroup() %>%
  mutate(accommodates = ifelse(is.na(accommodates), mean(accommodates, na.rm = TRUE), accommodates),
         availability_90 = ifelse(is.na(availability_90), mean(availability_90, na.rm = TRUE), availability_90),
         guests_included = ifelse(is.na(guests_included), mean(guests_included, na.rm = TRUE), guests_included), 
         host_total_listings_count = ifelse(is.na(host_total_listings_count), mean(host_total_listings_count, na.rm = TRUE), host_total_listings_count),
         latitude = ifelse(is.na(latitude), mean(latitude, na.rm = TRUE), latitude),
         security_deposit = ifelse(is.na(security_deposit), mean(security_deposit, na.rm = TRUE), security_deposit),
         minimum_nights = ifelse(is.na(minimum_nights), mean(minimum_nights, na.rm = TRUE), minimum_nights),
         cancellation_policy = ifelse(cancellation_policy %in% c("strict", "super_strict_30"), "strict", cancellation_policy),
         cancellation_policy = as.factor(cancellation_policy))

#Variable Cleaning:-bedrooms,host_listings_count,host_response_time,neighborhood,
#property_category, square_feet,weekly_price

airbnb_train <- airbnb_train %>% 
mutate(bedrooms = ifelse(is.na(bedrooms), mean(bedrooms, na.rm=TRUE), bedrooms), 
       host_listings_count = as.factor(host_listings_count), 
       host_response_time =as.factor(ifelse(is.na(host_response_time), "missing", host_response_time)),
       neighborhood = as.factor(ifelse(neighborhood == "La Jolla Village", "La Jolla", neighborhood)), 
       property_category = as.factor(ifelse(property_type %in% c("Apartment", "Serviced apartment", "Loft"), "apartment", 
                                            ifelse(property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel"), "hotel", 
                                                   ifelse(property_type %in% c("Townhouse", "Condominium"), "condo", 
                                                          ifelse(property_type %in% c("Bungalow", "House"), "house","other"))))), 
       square_feet = ifelse(is.na(square_feet),mean(square_feet, na.rm=TRUE), square_feet), 
       weekly_price = ifelse(is.na(weekly_price),mean(weekly_price, na.rm=TRUE), weekly_price)) %>%
  select(-country) %>%
  group_by(neighborhood) %>%
  mutate(market = if_else(is.na(market), na.omit(market)[1], market)) %>%
  ungroup() 

#Variable Cleaning:-availability_60,beds,country_code,first_review,host_since,
#maximum_nights,neighborhood_group,room_type,state,zipcode

airbnb_train$availability_60[is.na(airbnb_train$availability_60)] <- mean(airbnb_train$availability_60,na.rm=TRUE)
airbnb_train$availability_360[is.na(airbnb_train$availability_360)] <- mean(airbnb_train$availability_360,na.rm=TRUE)
airbnb_train$beds[is.na(airbnb_train$beds)] <- mean(airbnb_train$beds, na.rm=TRUE)
airbnb_train$country_code[is.na(airbnb_train$country_code)] <- "Unknown"
airbnb_train$first_review[is.na(airbnb_train$first_review)] <- "01/01/1999"
airbnb_train$host_since[is.na(airbnb_train$host_since)] <- "01/01/1999"
airbnb_train$maximum_nights[is.na(airbnb_train$maximum_nights)] <- mean(airbnb_train$maximum_nights, na.rm=TRUE)
airbnb_train$neighborhood_group[is.na(airbnb_train$neighborhood_group)] <- "Unknown"
airbnb_train$room_type[is.na(airbnb_train$room_type)] <- "Unknown"
airbnb_train$state <- toupper(airbnb_train$state)
unique_state <- unique(airbnb_train$state)
airbnb_train$state[is.na(airbnb_train$state)] <- "Unknown"
airbnb_train$zipcode[is.na(airbnb_train$zipcode)] <- "Unknown"
airbnb_train$zipcode <- as.character(airbnb_train$zipcode)

#Variable Cleaning:-availability_30
airbnb_train$availability_30[is.na(airbnb_train$availability_30)] <- mean(airbnb_train$availability_30, na.rm = TRUE)
airbnb_train$availability_30 <- as.numeric(airbnb_train$availability_30)


#Variable Cleaning:-#bed_category for bed_type
airbnb_train<-airbnb_train%>%
  mutate(bed_category = if_else(bed_type == "Real Bed", "bed", "other"))%>%
  mutate(bed_category = as.factor(bed_category))


#Variable Cleaning:-#CLEANING_FEE
#Converting cleaning_fee to number and replacing NA's with 0
airbnb_train<-airbnb_train%>%
  mutate(cleaning_fee = as.numeric(gsub("[$,]", "", cleaning_fee)))%>%
  mutate(cleaning_fee = if_else(is.na(cleaning_fee),mean(cleaning_fee,na.rm=TRUE), cleaning_fee))


#Variable Cleaning:-CHARGES_FOR_EXTRA
#Create a new (factor) variable called "charges_for_extra" which has the value "YES" if extra_people > 0 and "NO" if extra_people is 0 or NA
airbnb_train <- airbnb_train %>%
  mutate(charges_for_extra = ifelse(extra_people > 0, "YES", "NO"),
         charges_for_extra = as.factor(charges_for_extra))



#Variable Cleaning:- Adding new features :- HOST_ACCEPTANCE_RATE and HOST_RESPONSE
#"ALL" if host_acceptance_rate = 100%, "SOME" if host_acceptance_rate < 100%, and "MISSING" if it's NA.
#"ALL" if host_response_rate = 100%, "SOME" if host_response_rate < 100%, and "MISSING" if it's NA.
airbnb_train <- airbnb_train %>%
  mutate(host_acceptance_rate = as.numeric(gsub("%", "", host_acceptance_rate)),
         host_response_rate = as.numeric(gsub("%", "", host_response_rate)),
         host_acceptance = case_when(
           host_acceptance_rate == 100 ~ "ALL",
           host_acceptance_rate < 100 ~ "SOME",
           is.na(host_acceptance_rate) ~ "MISSING"
         ),
         host_acceptance = as.factor(host_acceptance),
         host_response = case_when(
           host_response_rate == 100 ~ "ALL",
           host_response_rate < 100 ~ "SOME",
           is.na(host_response_rate) ~ "MISSING"
         ),
         host_response = as.factor(host_response))

#Remove old names of features
airbnb_train<-airbnb_train%>%select(-extra_people,host_response_rate,host_acceptance_rate)

# Convert house_rules to lowercase for case-insensitive matching
house_rules_lower <- tolower(airbnb_train$house_rules)

# Check for the presence of keywords for smoking and pets rules
has_smoking_allowed <- ifelse(grepl("smoking", house_rules_lower) & grepl("allowed", house_rules_lower), "Yes", "No")
has_smoking_no <- ifelse(grepl("smoking", house_rules_lower) & grepl("no", house_rules_lower), "Yes", "No")

has_pets_allowed <- ifelse(grepl("pets", house_rules_lower) & grepl("allowed", house_rules_lower), "Yes", "No")
has_pets_no <- ifelse(grepl("pets", house_rules_lower) & grepl("no", house_rules_lower), "Yes", "No")

# Combine the rules into a single binary variable indicating presence of rules
has_rules <- ifelse(has_smoking_allowed == "Yes" | has_smoking_no == "Yes" | has_pets_allowed == "Yes" | has_pets_no == "Yes", "Yes", "No")

# Replace NA values with "No rules"
has_rules[is.na(has_rules)] <- "No rules"

# Add the new variable to the dataset
airbnb_train$has_rules <- has_rules

# View the updated dataset
head(airbnb_train)

#name column:just checking for duplicate values 
null_name_indices <- which(is.na(airbnb_train$name))

# Replace null values with "No Name"
airbnb_train$name[null_name_indices] <- "No Name"


#Variables Cleaning :PRICE VARIABLE:converting to numerical and replacing NA with 0
airbnb_train<-airbnb_train%>%
  mutate(price = as.numeric(gsub("[$,]", "", price)),
         price = if_else(is.na(price), mean(price,na.rm=TRUE), price))

library(tidytext)
#Variable Cleaning:SPACE VARIABLE:
space_words <- airbnb_train %>%
  filter(!is.na(space)) %>%  # Remove rows with NA values in 'space'
  unnest_tokens(word, space) %>%
  count(word, sort = TRUE)

# Calculate total words
total_words <- space_words %>%
  summarize(total = sum(n))

# Remove stop words
space_words_filtered <- anti_join(space_words, stop_words, by = c("word" = "word"))

# Remove numbers
space_words_filtered <- space_words_filtered %>%
  filter(!grepl("\\d", word))

# Create a new column indicating if any of the keywords are present
keywords <- c("balcony", "bedroom", "kitchen", "bathroom")  # Example keywords
space_words_filtered <- space_words_filtered %>%
  mutate(has_space_features = ifelse(word %in% keywords, "Yes", "No"))

# Handle missing values and set them to "No space features"
space_words_filtered$has_space_features[is.na(space_words_filtered$has_space_features)] <- "No space features"


airbnb_train <- airbnb_train %>%
  mutate(has_space_features = ifelse(space %in% space_words_filtered$word, "Yes", "No"))

# Handle missing values and set them to "No space features"
airbnb_train$has_space_features[is.na(airbnb_train$has_space_features)] <- "No space features"
#______________________________TEST DATA__________________#
# Function to tokenize, normalize, and create binary indicators for a column
process_column_test <- function(data, column_name) {
  # Tokenize the column by splitting on commas
  data <- data %>%
    mutate(
      !!sym(column_name) := str_split(!!sym(column_name), ",")
    )
  
  # Normalize the column by converting to lowercase and removing special characters
  data <- data %>%
    mutate(
      !!sym(column_name) := lapply(!!sym(column_name), function(x) tolower(gsub("[[:punct:]]", "", x)))
    )
  
  # Remove leading and trailing whitespace
  data <- data %>%
    mutate(
      !!sym(column_name) := lapply(!!sym(column_name), function(x) trimws(x))
    )
  
  # Extract unique values from the column
  unique_values <- unique(unlist(data[[column_name]]))
  
  # Create binary indicators for each unique value
  binary_indicators <- sapply(unique_values, function(value) {
    sapply(data[[column_name]], function(value_list) {
      as.numeric(value %in% value_list)
    })
  })
  
  # Convert the binary indicators to a data frame
  binary_df <- as.data.frame(binary_indicators)
  
  # Prefix column names with 'has_' and ensure uniqueness
  colnames(binary_df) <- make.unique(paste0("has_", unique_values))
  
  # Bind the binary indicators to the original data frame
  data <- cbind(data, binary_df)
  
  return(data)
}
#Variable Cleaning:-experiences_offered,smart_location,host_neighbourhood,bathrooms,
#monthly_price, license_status,host_verifications, amenities
airbnb_test <- test_x %>%
  mutate(
    experiences_offered = as.factor(experiences_offered),
    smart_location = as.factor(smart_location),
    host_neighbourhood =ifelse(is.na(host_neighbourhood),"Not Mentioned", host_neighbourhood),
    host_neighbourhood = as.factor(host_neighbourhood),
    bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
    monthly_price = ifelse(is.na(monthly_price),mean(monthly_price,na.rm=TRUE), monthly_price),
    license_status = case_when(
      is.na(license) ~ "No",
      grepl("pending", license, ignore.case = TRUE) ~ "Pending",
      grepl("^\\d+$", license) ~ "Yes",
      TRUE ~ "Yes"
    ),
    license_status = as.factor(license_status)
  ) %>%
  mutate(
    city = trimws(tolower(city)),
    city = gsub("[[:punct:]]", "", city),
    city = gsub("\\s+", " ", city),
    city = ifelse(is.na(city), "Missing", city), # Handling missing values
    city = as.factor(city) 
  ) %>%
  select(-license)
airbnb_test <- airbnb_test %>%
  process_column_test("host_verifications") %>%
  process_column_test("amenities")

# Remove the host_verifications column from airbnb_test
airbnb_test <- airbnb_test[, !names(airbnb_test) %in% c("host_verifications")]

# Remove the amenities column from airbnb_test
airbnb_test <- airbnb_test[, !names(airbnb_test) %in% c("amenities")]

#Variable Cleaning:-accommodates,availability_90,guests_included,host_total_listings_count,
#latitude, security_deposit,minimum_nights, cancellation_policy
airbnb_test <- airbnb_test %>%
  ungroup() %>%
  mutate(accommodates = ifelse(is.na(accommodates), mean(accommodates, na.rm = TRUE), accommodates),
         availability_90 = ifelse(is.na(availability_90), mean(availability_90, na.rm = TRUE), availability_90),
         guests_included = ifelse(is.na(guests_included), mean(guests_included, na.rm = TRUE), guests_included), 
         host_total_listings_count = ifelse(is.na(host_total_listings_count), mean(host_total_listings_count, na.rm = TRUE), host_total_listings_count),
         latitude = ifelse(is.na(latitude), mean(latitude, na.rm = TRUE), latitude),
         security_deposit = ifelse(is.na(security_deposit), mean(security_deposit, na.rm = TRUE), security_deposit),
         minimum_nights = ifelse(is.na(minimum_nights), mean(minimum_nights, na.rm = TRUE), minimum_nights),
         cancellation_policy = ifelse(cancellation_policy %in% c("strict", "super_strict_30"), "strict", cancellation_policy),
         cancellation_policy = as.factor(cancellation_policy))

#Variable Cleaning:-bedrooms,host_listings_count,host_response_time,neighborhood,
#property_category, square_feet,weekly_price

airbnb_test <- airbnb_test %>% 
  mutate(bedrooms = ifelse(is.na(bedrooms), mean(bedrooms, na.rm=TRUE), bedrooms), 
         host_listings_count = as.factor(host_listings_count), 
         host_response_time =as.factor(ifelse(is.na(host_response_time), "missing", host_response_time)),
         neighborhood = as.factor(ifelse(neighborhood == "La Jolla Village", "La Jolla", neighborhood)), 
         property_category = as.factor(ifelse(property_type %in% c("Apartment", "Serviced apartment", "Loft"), "apartment", 
                                              ifelse(property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel"), "hotel", 
                                                     ifelse(property_type %in% c("Townhouse", "Condominium"), "condo", 
                                                            ifelse(property_type %in% c("Bungalow", "House"), "house","other"))))), 
         square_feet = ifelse(is.na(square_feet),mean(square_feet,na.rm=TRUE), square_feet), 
         weekly_price = ifelse(is.na(weekly_price),mean(weekly_price,na.rm=TRUE), weekly_price)) %>%
  select(-country) %>%
  group_by(neighborhood) %>%
  mutate(market = if_else(is.na(market), na.omit(market)[1], market)) %>%
  ungroup()  
#Variable Cleaning:-availability_60,beds,country_code,first_review,host_since,
#maximum_nights,neighborhood_group,room_type,state,zipcode

airbnb_test$availability_60[is.na(airbnb_test$availability_60)] <- mean(airbnb_test$availability_60,na.rm=TRUE)
airbnb_test$availability_360[is.na(airbnb_test$availability_360)] <- mean(airbnb_test$availability_360,na.rm=TRUE)
airbnb_test$beds[is.na(airbnb_test$beds)] <- mean(airbnb_test$beds, na.rm=TRUE)
airbnb_test$country_code[is.na(airbnb_test$country_code)] <- "Unknown"
airbnb_test$first_review[is.na(airbnb_test$first_review)] <- "01/01/1999"
airbnb_test$host_since[is.na(airbnb_test$host_since)] <- "01/01/1999"
airbnb_test$maximum_nights[is.na(airbnb_test$maximum_nights)] <- mean(airbnb_test$maximum_nights, na.rm=TRUE)
airbnb_test$neighborhood_group[is.na(airbnb_test$neighborhood_group)] <- "Unknown"
airbnb_test$room_type[is.na(airbnb_test$room_type)] <- "Unknown"
airbnb_test$state <- toupper(airbnb_test$state)
unique_state <- unique(airbnb_test$state)
airbnb_test$state[is.na(airbnb_test$state)] <- "Unknown"
airbnb_test$zipcode[is.na(airbnb_test$zipcode)] <- "Unknown"
airbnb_test$zipcode <- as.character(airbnb_test$zipcode)

#availability_30
airbnb_test$availability_30[is.na(airbnb_test$availability_30)] <- mean(airbnb_test$availability_30, na.rm = TRUE)
airbnb_test$availability_30 <- as.numeric(airbnb_test$availability_30)

#bed_category for bed_type
airbnb_test<-airbnb_test%>%
  mutate(bed_category = if_else(bed_type == "Real Bed", "bed", "other"))%>%
  mutate(bed_category = as.factor(bed_category))

#CLEANING_FEE
#converting cleaning_fee to number and replacing NA's with 0
airbnb_test<-airbnb_test%>%
  mutate(cleaning_fee = as.numeric(gsub("[$,]", "", cleaning_fee)))%>%
  mutate(cleaning_fee = if_else(is.na(cleaning_fee),mean(cleaning_fee,na.rm=TRUE), cleaning_fee))

#CHARGES_FOR_EXTRA
#Create a new (factor) variable called "charges_for_extra" which has the value "YES" if extra_people > 0 and "NO" if extra_people is 0 or NA
airbnb_test <- airbnb_test %>%
  mutate(charges_for_extra = ifelse(extra_people > 0, "YES", "NO"),
         charges_for_extra = as.factor(charges_for_extra))


#HOST_ACCEPTANCE_RATE and HOST_RESPONSE
#Create a new (factor) variable called "host_acceptance" from host_acceptance_rate with the values 
#"ALL" if host_acceptance_rate = 100%, "SOME" if host_acceptance_rate < 100%, and "MISSING" if it's NA.
#Similarly, create a new (factor) variable called "host_response" with the values 
#"ALL" if host_response_rate = 100%, "SOME" if host_response_rate < 100%, and "MISSING" if it's NA.
airbnb_test <- airbnb_test %>%
  mutate(host_acceptance_rate = as.numeric(gsub("%", "", host_acceptance_rate)),
         host_response_rate = as.numeric(gsub("%", "", host_response_rate)),
         host_acceptance = case_when(
           host_acceptance_rate == 100 ~ "ALL",
           host_acceptance_rate < 100 ~ "SOME",
           is.na(host_acceptance_rate) ~ "MISSING"
         ),
         host_acceptance = as.factor(host_acceptance),
         host_response = case_when(
           host_response_rate == 100 ~ "ALL",
           host_response_rate < 100 ~ "SOME",
           is.na(host_response_rate) ~ "MISSING"
         ),
         host_response = as.factor(host_response))
airbnb_test<-airbnb_test%>%select(-extra_people,host_response_rate,host_acceptance_rate)



#HOUSE_RULES VARIABLE
# Convert house_rules to lowercase for case-insensitive matching
house_rules_lower <- tolower(airbnb_test$house_rules)

# Check for the presence of keywords for smoking and pets rules
has_smoking_allowed <- ifelse(grepl("smoking", house_rules_lower) & grepl("allowed", house_rules_lower), "Yes", "No")
has_smoking_no <- ifelse(grepl("smoking", house_rules_lower) & grepl("no", house_rules_lower), "Yes", "No")

has_pets_allowed <- ifelse(grepl("pets", house_rules_lower) & grepl("allowed", house_rules_lower), "Yes", "No")
has_pets_no <- ifelse(grepl("pets", house_rules_lower) & grepl("no", house_rules_lower), "Yes", "No")

# Combine the rules into a single binary variable indicating presence of rules
has_rules <- ifelse(has_smoking_allowed == "Yes" | has_smoking_no == "Yes" | has_pets_allowed == "Yes" | has_pets_no == "Yes", "Yes", "No")

# Replace NA values with "No rules"
has_rules[is.na(has_rules)] = "No rules"

# Add the new variable to the dataset
airbnb_test$has_rules <- has_rules

# View the updated dataset
head(airbnb_test)


#name column:just checking for duplicate values 
null_name_indices <- which(is.na(airbnb_test$name))

# Replace null values with "No Name"
airbnb_test$name[null_name_indices] <- "No Name"


#PRICE VARIABLE:converting to numerical and replacing NA with mean
airbnb_test<-airbnb_test%>%
  mutate(price = as.numeric(gsub("[$,]", "", price)),
         price = if_else(is.na(price),mean(price, na.rm=TRUE), price))

#SPACE VARIABLE:
space_words <- airbnb_test %>%
  filter(!is.na(space)) %>%  # Remove rows with NA values in 'space'
  unnest_tokens(word, space) %>%
  count(word, sort = TRUE)

# Calculate total words
total_words <- space_words %>%
  summarize(total = sum(n))

# Remove stop words
space_words_filtered <- anti_join(space_words, stop_words, by = c("word" = "word"))

# Remove numbers
space_words_filtered <- space_words_filtered %>%
  filter(!grepl("\\d", word))

# Create a new column indicating if any of the keywords are present
keywords <- c("balcony", "bedroom", "kitchen", "bathroom")  # Example keywords
space_words_filtered <- space_words_filtered %>%
  mutate(has_space_features = ifelse(word %in% keywords, "Yes", "No"))

# Handle missing values and set them to "No space features"
space_words_filtered$has_space_features[is.na(space_words_filtered$has_space_features)] <- "No space features"

airbnb_test <- airbnb_test %>%
  mutate(has_space_features = ifelse(space %in% space_words_filtered$word, "Yes", "No"))

# Handle missing values and set them to "No space features"
airbnb_test$has_space_features[is.na(airbnb_test$has_space_features)] <- "No space features"



# Check for missing values in each variable
missing_values <- colSums(is.na(airbnb_test))

# Print the names of variables with missing values
names_with_missing <- names(missing_values[missing_values > 0])
print(names_with_missing)


airbnb_train$space[is.na(airbnb_train$space)] <- "No space features provided"  # Replace with a default value
airbnb_train$house_rules[is.na(airbnb_train$house_rules)] <- "No house rules specified"  # Replace with a default value
airbnb_train$host_location[is.na(airbnb_train$host_location)] <- "Unknown"  # Replace with a default value
airbnb_train$host_response_rate[is.na(airbnb_train$host_response_rate)] <- 0  # Replace with 0
airbnb_train$host_acceptance_rate[is.na(airbnb_train$host_acceptance_rate)] <- 0  # Replace with 0
airbnb_train$market[is.na(airbnb_train$market)] <- "Unknown"  # Replace with a default value
airbnb_train$property_type[is.na(airbnb_train$property_type)] <- "Other"  # Replace with a default value
airbnb_train$jurisdiction_names[is.na(airbnb_train$jurisdiction_names)] <- "Unknown"  # Replace with a default value

airbnb_test$space[is.na(airbnb_test$space)] <- "No space features provided"  # Replace with a default value
airbnb_test$house_rules[is.na(airbnb_test$house_rules)] <- "No house rules"  # Replace with a default value
airbnb_test$host_location[is.na(airbnb_test$host_location)] <- "Unknown"  # Replace with a default value
airbnb_test$host_response_rate[is.na(airbnb_test$host_response_rate)] <- 0  # Replace with 0
airbnb_test$host_acceptance_rate[is.na(airbnb_test$host_acceptance_rate)] <- 0  # Replace with 0
airbnb_test$market[is.na(airbnb_test$market)] <- "Unknown"  # Replace with a default value
airbnb_test$property_type[is.na(airbnb_test$property_type)] <- "Other"  # Replace with a default value
airbnb_test$jurisdiction_names[is.na(airbnb_test$jurisdiction_names)] <- "Unknown"  # Replace with a default value

# Check for missing values in each variable
missing_values <- colSums(is.na(airbnb_test))

# Print the names of variables with missing values
names_with_missing <- names(missing_values[missing_values > 0])
print(names_with_missing)

# Access variable
clean_access <- function(access) {
  access <- tolower(access)  # Standardize the text by converting to lowercase
  data_frame(
    parking = as.integer(str_detect(access, "parking")),
    pets_allowed = as.integer(str_detect(access, "pets allowed|pets live on this property")),
    private_entrance = as.integer(str_detect(access, "private entrance")),
    kitchen_access = as.integer(str_detect(access, "kitchen")),
    smoke_free = as.integer(str_detect(access, "no smoking|smoke free")),
    event_friendly = as.integer(str_detect(access, "suitable for events")),
    full_access = as.integer(str_detect(access, "all of the home is accessible|full access"))
  )
}

airbnb_train <- airbnb_train %>%
  rowwise() %>%
  mutate(cleaned_access = list(clean_access(access))) %>%
  ungroup() %>%
  bind_cols()

airbnb_train <- airbnb_train %>%
  tidyr::unnest(cleaned_access)

airbnb_test <- airbnb_test %>%
  rowwise() %>%
  mutate(cleaned_access = list(clean_access(access))) %>%
  ungroup() %>%
  bind_cols()

airbnb_test <- airbnb_test %>%
  tidyr::unnest(cleaned_access)

# Host_location variable
categorize_location <- function(location) {
  location <- tolower(location)  # Convert to lowercase to standardize the input
  case_when(
    str_detect(location, "northeast|new york|massachusetts|new jersey|connecticut|pennsylvania") ~ "Northeast US",
    str_detect(location, "southeast|florida|georgia|north carolina|south carolina") ~ "Southeast US",
    str_detect(location, "midwest|illinois|ohio|michigan|indiana|wisconsin|missouri") ~ "Midwest US",
    str_detect(location, "southwest|texas|arizona|new mexico|oklahoma") ~ "Southwest US",
    str_detect(location, "west|california|nevada|oregon|washington|colorado") ~ "West US",
    str_detect(location, "canada|europe|asia|africa|australia|latin america") ~ "International",
    TRUE ~ "Other"  # Default case if none of the above matches
  )
}

airbnb_train <- airbnb_train %>%
  mutate(region = categorize_location(host_location))

airbnb_train <- airbnb_train %>%
  select(-host_location)

airbnb_test <- airbnb_test %>%
  mutate(region = categorize_location(host_location))

airbnb_test <- airbnb_test %>%
  select(-host_location)

# Jurisdiction_names variable
categorize_jurisdiction_by_state <- function(jurisdiction) {
  jurisdiction <- tolower(jurisdiction)  # Convert to lowercase to standardize
  case_when(
    str_detect(jurisdiction, "california|los angeles|malibu|santa monica|palo alto") ~ "California",
    str_detect(jurisdiction, "louisiana|new orleans") ~ "Louisiana",
    str_detect(jurisdiction, "washington|district of columbia") ~ "Washington DC",
    str_detect(jurisdiction, "oregon|multnomah|portland|lane county|washington county") ~ "Oregon",
    str_detect(jurisdiction, "illinois|chicago|cook county|oak park") ~ "Illinois",
    str_detect(jurisdiction, "arkansas") ~ "Arkansas",
    str_detect(jurisdiction, "connecticut") ~ "Connecticut",
    str_detect(jurisdiction, "colorado") ~ "Colorado",
    str_detect(jurisdiction, "montgomery county") & str_detect(jurisdiction, "md") ~ "Maryland",
    TRUE ~ "Other"  # Default category
  )
}

airbnb_train <- airbnb_train %>%
  mutate(jurisdiction_state = categorize_jurisdiction_by_state(jurisdiction_names),
         jurisdiction_state=as.factor(jurisdiction_state))

airbnb_train <- airbnb_train %>%
  select(-jurisdiction_names)

airbnb_test <- airbnb_test %>%
  mutate(jurisdiction_state = categorize_jurisdiction_by_state(jurisdiction_names),
         jurisdiction_state=as.factor(jurisdiction_state))

airbnb_test <- airbnb_test %>%
  select(-jurisdiction_names)
#----------------------------TEXT MINING------------------------#
 
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(tm::stopwords(kind="en")) %>% #remove stopwords
    #stemDocument %>%
    word_tokenizer 
}

## description variable 
it_train<- itoken(airbnb_train$description, 
                   preprocessor = tolower, 
                   tokenizer = cleaning_tokenizer, 
                   ids = rownames(airbnb_train), 
                   progressbar = FALSE)

vocab <- create_vocabulary(it_train)
vocab_small <- prune_vocabulary(vocab, vocab_term_max = 500)
vectorizer <- vocab_vectorizer(vocab_small)
dtm_train <- create_dtm(it_train, vectorizer)
dim(dtm_train)
Matrix::colSums(dtm_train)

airbnb_train <- airbnb_train %>%
  mutate(desc_category = case_when(grepl("private", description, ignore.case = TRUE) ~ "private",
                                   grepl("kitchen", description, ignore.case = TRUE) ~ "kitchen",
                                   grepl("restaurants", description, ignore.case = TRUE) ~ "restaurants",
                                   grepl("access", description, ignore.case = TRUE) ~ "access",
                                   grepl("space", description, ignore.case = TRUE) ~ "space",
                                   grepl("full", description, ignore.case = TRUE) ~ "full",
                                   grepl("walk", description, ignore.case = TRUE) ~ "walk",
                                   grepl("great", description, ignore.case = TRUE) ~ "great",
                                   grepl("park", description, ignore.case = TRUE) ~ "park",
                                   grepl("new", description, ignore.case = TRUE) ~ "new", TRUE ~ "other"),
         desc_category = as.factor(desc_category))

## neighborhood overview & street 
it_train2 <- itoken(airbnb_train$neighborhood_overview, 
                    preprocessor = tolower, 
                    tokenizer = cleaning_tokenizer, 
                    ids = rownames(airbnb_train), 
                    progressbar = FALSE)

vocab2 <- create_vocabulary(it_train2)
vocab_small2 <- prune_vocabulary(vocab2, vocab_term_max = 500)
vectorizer2 <- vocab_vectorizer(vocab_small2)
dtm_train2 <- create_dtm(it_train2, vectorizer2)
dim(dtm_train2)
Matrix::colSums(dtm_train2)

airbnb_train <- airbnb_train %>%
  mutate(neighborhood_overview = ifelse(is.na(neighborhood_overview), "No Neighborhood Overview", neighborhood_overview),
         neighborhood_overview_category = case_when(grepl("restaurants", neighborhood_overview, ignore.case = TRUE) ~ "restaurants",
                                                    grepl("park", neighborhood_overview, ignore.case = TRUE) ~ "park",
                                                    grepl("walk", neighborhood_overview, ignore.case = TRUE) ~ "walk",
                                                    grepl("great", neighborhood_overview, ignore.case = TRUE) ~ "great",
                                                    grepl("bars", neighborhood_overview, ignore.case = TRUE) ~ "bars",
                                                    grepl("walking", neighborhood_overview, ignore.case = TRUE) ~ "walking",
                                                    grepl("blocks", neighborhood_overview, ignore.case = TRUE) ~ "blocks",
                                                    grepl("shops", neighborhood_overview, ignore.case = TRUE) ~ "shops",
                                                    grepl("city", neighborhood_overview, ignore.case = TRUE) ~ "city",
                                                    grepl("downtown", neighborhood_overview, ignore.case = TRUE) ~ "downtown", TRUE ~ "other"),
         neighborhood_overview_category = as.factor(neighborhood_overview_category),
         street_extract = sub(",.*", "", street))

## cleaning tokenizer 
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(tm::stopwords(kind="en")) %>% #remove stopwords
    #stemDocument %>%
    word_tokenizer 
}

## description variable 
it_test <- itoken(airbnb_test$description, 
                   preprocessor = tolower, 
                   tokenizer = cleaning_tokenizer, 
                   ids = rownames(airbnb_test), 
                   progressbar = FALSE)

vocab <- create_vocabulary(it_test)
vocab_small <- prune_vocabulary(vocab, vocab_term_max = 500)
vectorizer <- vocab_vectorizer(vocab_small)
dtm_test <- create_dtm(it_train, vectorizer)
dim(dtm_test)
Matrix::colSums(dtm_test)

airbnb_test <- airbnb_test %>%
  mutate(desc_category = case_when(grepl("private", description, ignore.case = TRUE) ~ "private",
                                   grepl("kitchen", description, ignore.case = TRUE) ~ "kitchen",
                                   grepl("restaurants", description, ignore.case = TRUE) ~ "restaurants",
                                   grepl("access", description, ignore.case = TRUE) ~ "access",
                                   grepl("space", description, ignore.case = TRUE) ~ "space",
                                   grepl("full", description, ignore.case = TRUE) ~ "full",
                                   grepl("walk", description, ignore.case = TRUE) ~ "walk",
                                   grepl("great", description, ignore.case = TRUE) ~ "great",
                                   grepl("park", description, ignore.case = TRUE) ~ "park",
                                   grepl("new", description, ignore.case = TRUE) ~ "new", TRUE ~ "other"),
         desc_category = as.factor(desc_category))

## neighborhood overview & street 
it_test2 <- itoken(airbnb_test$neighborhood_overview, 
                    preprocessor = tolower, 
                    tokenizer = cleaning_tokenizer, 
                    ids = rownames(airbnb_test), 
                    progressbar = FALSE)

vocab2 <- create_vocabulary(it_test2)
vocab_small2 <- prune_vocabulary(vocab2, vocab_term_max = 500)
vectorizer2 <- vocab_vectorizer(vocab_small2)
dtm_test2 <- create_dtm(it_test2, vectorizer2)
dim(dtm_test2)
Matrix::colSums(dtm_test2)

airbnb_test <- airbnb_test %>%
  mutate(neighborhood_overview = ifelse(is.na(neighborhood_overview), "No Neighborhood Overview", neighborhood_overview),
         neighborhood_overview_category = case_when(grepl("restaurants", neighborhood_overview, ignore.case = TRUE) ~ "restaurants",
                                                    grepl("park", neighborhood_overview, ignore.case = TRUE) ~ "park",
                                                    grepl("walk", neighborhood_overview, ignore.case = TRUE) ~ "walk",
                                                    grepl("great", neighborhood_overview, ignore.case = TRUE) ~ "great",
                                                    grepl("bars", neighborhood_overview, ignore.case = TRUE) ~ "bars",
                                                    grepl("walking", neighborhood_overview, ignore.case = TRUE) ~ "walking",
                                                    grepl("blocks", neighborhood_overview, ignore.case = TRUE) ~ "blocks",
                                                    grepl("shops", neighborhood_overview, ignore.case = TRUE) ~ "shops",
                                                    grepl("city", neighborhood_overview, ignore.case = TRUE) ~ "city",
                                                    grepl("downtown", neighborhood_overview, ignore.case = TRUE) ~ "downtown", TRUE ~ "other"),
         neighborhood_overview_category = as.factor(neighborhood_overview_category),
         street_extract = sub(",.*", "", street)) 


## cleaning tokenizer 
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(tm::stopwords(kind="en")) %>% #remove stopwords
    #stemDocument %>%
    word_tokenizer 
}

## description variable 
it_train <- itoken(airbnb_train$host_about, 
                   preprocessor = tolower, 
                   tokenizer = cleaning_tokenizer, 
                   ids = rownames(airbnb_train), 
                   progressbar = FALSE)

vocab <- create_vocabulary(it_train)
vocab_small <- prune_vocabulary(vocab, vocab_term_max = 500)
vectorizer <- vocab_vectorizer(vocab_small)
dtm_train <- create_dtm(it_train, vectorizer)
dim(dtm_train)
Matrix::colSums(dtm_train)
## description variable 
it_test <- itoken(airbnb_test$host_about, 
                  preprocessor = tolower, 
                  tokenizer = cleaning_tokenizer, 
                  ids = rownames(airbnb_test), 
                  progressbar = FALSE)

vocab_test <- create_vocabulary(it_test)
vocab_small_test <- prune_vocabulary(vocab_test, vocab_term_max = 500)
vectorizer_test <- vocab_vectorizer(vocab_small_test)
dtm_test <- create_dtm(it_test, vectorizer)
dim(dtm_test)
Matrix::colSums(dtm_test)

# Define categories using regex patterns
categories <- list(
  Travel_Locations = "travel|exploring|adventure|places|beach|city|san|york|francisco|la|nyc|austin|angeles|boston|chicago|seattle|portland|diego|nashville|dc|california|europe|paris|london|mexico|france|spain|italy",
  Home_Living = "home|house|apartment|cozy|living|stay|comfortable|space|room|property|furnished|interior",
  Social_Entertainment = "music|art|movies|books|entertainment|sports|dancing|shows|film|theater|concerts|festivals|party",
  Food_Cooking = "food|cooking|eating|cuisine|restaurants|coffee|dining|wine|dinner|meals|cook|chef",
  Outdoor_Adventure = "hiking|outdoors|beach|park|nature|garden|mountains|walking|biking|running|fishing|camping",
  Professional_Business = "business|professional|work|job|company|industry|office|career|entrepreneur|management",
  Family_Relationships = "family|friends|kids|husband|wife|daughter|son|guests|host|meeting|community|social",
  Culture_Lifestyle = "culture|lifestyle|history|tradition|heritage|fashion|style|design|modern|vintage|arts",
  Health_Wellbeing = "health|wellness|fitness|yoga|meditation|exercise|gym|sports|diet|healthy",
  Education_Learning = "learning|education|school|university|classes|courses|teaching|teacher|student|training|knowledge"
)
airbnb_train <- airbnb_train %>%
  mutate(host_mentions = case_when(
    grepl(categories$Travel_Locations, tolower(host_about), perl = TRUE) ~ "Travel and Locations",
    grepl(categories$Home_Living, tolower(host_about), perl = TRUE) ~ "Home and Living",
    grepl(categories$Social_Entertainment, tolower(host_about), perl = TRUE) ~ "Social and Entertainment",
    grepl(categories$Food_Cooking, tolower(host_about), perl = TRUE) ~ "Food and Cooking",
    grepl(categories$Outdoor_Adventure, tolower(host_about), perl = TRUE) ~ "Outdoor and Adventure",
    grepl(categories$Professional_Business, tolower(host_about), perl = TRUE) ~ "Professional and Business",
    grepl(categories$Family_Relationships, tolower(host_about), perl = TRUE) ~ "Family and Relationships",
    grepl(categories$Culture_Lifestyle, tolower(host_about), perl = TRUE) ~ "Culture and Lifestyle",
    grepl(categories$Health_Wellbeing, tolower(host_about), perl = TRUE) ~ "Health and Well-being",
    grepl(categories$Education_Learning, tolower(host_about), perl = TRUE) ~ "Education and Learning",
    TRUE ~ "Other"
  ))

airbnb_train <- airbnb_train %>%
  mutate(host_mentions = as.factor(host_mentions))




airbnb_test <- airbnb_test%>%
  mutate(host_mentions = case_when(
    grepl(categories$Travel_Locations, tolower(host_about), perl = TRUE) ~ "Travel and Locations",
    grepl(categories$Home_Living, tolower(host_about), perl = TRUE) ~ "Home and Living",
    grepl(categories$Social_Entertainment, tolower(host_about), perl = TRUE) ~ "Social and Entertainment",
    grepl(categories$Food_Cooking, tolower(host_about), perl = TRUE) ~ "Food and Cooking",
    grepl(categories$Outdoor_Adventure, tolower(host_about), perl = TRUE) ~ "Outdoor and Adventure",
    grepl(categories$Professional_Business, tolower(host_about), perl = TRUE) ~ "Professional and Business",
    grepl(categories$Family_Relationships, tolower(host_about), perl = TRUE) ~ "Family and Relationships",
    grepl(categories$Culture_Lifestyle, tolower(host_about), perl = TRUE) ~ "Culture and Lifestyle",
    grepl(categories$Health_Wellbeing, tolower(host_about), perl = TRUE) ~ "Health and Well-being",
    grepl(categories$Education_Learning, tolower(host_about), perl = TRUE) ~ "Education and Learning",
    TRUE ~ "Other"
  ))

airbnb_test <- airbnb_test %>%
  mutate(host_mentions = as.factor(host_mentions))

## cleaning tokenizer 
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(tm::stopwords(kind="en")) %>% #remove stopwords
    #stemDocument %>%
    word_tokenizer 
}
## transit variable 
it_train <- itoken(airbnb_train$transit, 
                  preprocessor = tolower, 
                  tokenizer = cleaning_tokenizer, 
                  ids = rownames(airbnb_train), 
                  progressbar = FALSE)

vocab <- create_vocabulary(it_train)
vocab_small <- prune_vocabulary(vocab, vocab_term_max = 500)
vectorizer <- vocab_vectorizer(vocab_small)
dtm_train <- create_dtm(it_train, vectorizer)
dim(dtm_train)
Matrix::colSums(dtm_train)

airbnb_train <- airbnb_train %>%
  mutate(transit_category = case_when(grepl("bus ", transit, ignore.case = TRUE) ~ "bus ",
                                      grepl("walk", transit, ignore.case = TRUE) ~ "walk",
                                      grepl("train", transit, ignore.case = TRUE) ~ "train",
                                      grepl("downtown", transit, ignore.case = TRUE) ~ "downtown",
                                      grepl("metro", transit, ignore.case = TRUE) ~ "metro",
                                      grepl("subway", transit, ignore.case = TRUE) ~ "subway",
                                      grepl("station", transit, ignore.case = TRUE) ~ "station",
                                      grepl("airport", transit, ignore.case = TRUE) ~ "airport",
                                      grepl("public", transit, ignore.case = TRUE) ~ "public",
                                      grepl("min", transit, ignore.case = TRUE) ~ "min", TRUE ~ "other"),
         transit_category = as.factor(transit_category))

it_test <- itoken(airbnb_test$transit, 
                  preprocessor = tolower, 
                  tokenizer = cleaning_tokenizer, 
                  ids = rownames(airbnb_test), 
                  progressbar = FALSE)

vocab <- create_vocabulary(it_test)
vocab_small <- prune_vocabulary(vocab, vocab_term_max = 500)
vectorizer <- vocab_vectorizer(vocab_small)
dtm_test <- create_dtm(it_test, vectorizer)
dim(dtm_test)
Matrix::colSums(dtm_test)

airbnb_test <- airbnb_test %>%
  mutate(transit_category = case_when(grepl("bus ", transit, ignore.case = TRUE) ~ "bus ",
                                   grepl("walk", transit, ignore.case = TRUE) ~ "walk",
                                   grepl("train", transit, ignore.case = TRUE) ~ "train",
                                   grepl("downtown", transit, ignore.case = TRUE) ~ "downtown",
                                   grepl("metro", transit, ignore.case = TRUE) ~ "metro",
                                   grepl("subway", transit, ignore.case = TRUE) ~ "subway",
                                   grepl("station", transit, ignore.case = TRUE) ~ "station",
                                   grepl("airport", transit, ignore.case = TRUE) ~ "airport",
                                   grepl("public", transit, ignore.case = TRUE) ~ "public",
                                   grepl("min", transit, ignore.case = TRUE) ~ "min", TRUE ~ "other"),
         transit_category = as.factor(transit_category))

#Column: features
features_process <- function(data, column_name) {
  data <- data %>%
    mutate(
      !!sym(column_name) := str_split(!!sym(column_name), ",")
    )
  data <- data %>%
    mutate(
      !!sym(column_name) := lapply(!!sym(column_name), function(x) tolower(gsub("[[:punct:]]", "", x)))
    )
  data <- data %>%
    mutate(
      !!sym(column_name) := lapply(!!sym(column_name), function(x) trimws(x))
    )
  data[[column_name]] <- lapply(data[[column_name]], function(x) {
    x[is.na(x)] <- "missing_value_placeholder"
    x
  })
  unique_values <- unique(unlist(data[[column_name]]))
  binary_indicators <- sapply(unique_values, function(value) {
    sapply(data[[column_name]], function(value_list) {
      as.numeric(value %in% value_list)
    })
  })
  binary_df <- as.data.frame(binary_indicators)
  colnames(binary_df) <- make.unique(paste0("has_", unique_values))
  data <- cbind(data, binary_df)
  data <- data %>%
    select(-!!sym(column_name))
  
  return(data)
}

airbnb_train <- airbnb_train %>%
  features_process("features")

features_process_test <- function(data, column_name) {
  data <- data %>%
    mutate(
      !!sym(column_name) := str_split(!!sym(column_name), ",")
    )
  data <- data %>%
    mutate(
      !!sym(column_name) := lapply(!!sym(column_name), function(x) tolower(gsub("[[:punct:]]", "", x)))
    )
  data <- data %>%
    mutate(
      !!sym(column_name) := lapply(!!sym(column_name), function(x) trimws(x))
    )
  data[[column_name]] <- lapply(data[[column_name]], function(x) {
    x[is.na(x)] <- "missing_value_placeholder"
    x
  })
  unique_values <- unique(unlist(data[[column_name]]))
  binary_indicators <- sapply(unique_values, function(value) {
    sapply(data[[column_name]], function(value_list) {
      as.numeric(value %in% value_list)
    })
  })
  binary_df <- as.data.frame(binary_indicators)
  colnames(binary_df) <- make.unique(paste0("has_", unique_values))
  data <- cbind(data, binary_df)
  data <- data %>%
    select(-!!sym(column_name))
  
  return(data)
}


airbnb_test <- airbnb_test %>%
  features_process_test("features")


airbnb_train <- airbnb_train %>%
  mutate(interaction_category = case_when(grepl("available", description, ignore.case = TRUE) ~ "available",
                                          grepl("guests", description, ignore.case = TRUE) ~ "guests",
                                          grepl("need", description, ignore.case = TRUE) ~ "need",
                                          grepl("stay", description, ignore.case = TRUE) ~ "stay",
                                          grepl("questions", description, ignore.case = TRUE) ~ "questions",
                                          grepl("help", description, ignore.case = TRUE) ~ "help",
                                          grepl("happy", description, ignore.case = TRUE) ~ "happy",
                                          grepl("time", description, ignore.case = TRUE) ~ "time",
                                          grepl("anything", description, ignore.case = TRUE) ~ "anything",
                                          grepl("around", description, ignore.case = TRUE) ~ "around",
                                          grepl("email", description, ignore.case = TRUE) ~ "email",
                                          grepl("please", description, ignore.case = TRUE) ~ "please",
                                          grepl("answer", description, ignore.case = TRUE) ~ "answer",
                                          grepl("love", description, ignore.case = TRUE) ~ "love",
                                          grepl("call", description, ignore.case = TRUE) ~ "call",
                                          grepl("reccomendations", description, ignore.case = TRUE) ~ "reccomendations",
                                          grepl("provide", description, ignore.case = TRUE) ~ "provide",TRUE ~ "other"),
         interaction_category = as.factor(interaction_category))

#get rid of original interaction column in train 
airbnb_train<- airbnb_train%>%
  select(-interaction)


airbnb_test <- airbnb_test %>%
  mutate(interaction_category = case_when(grepl("available", description, ignore.case = TRUE) ~ "available",
                                          grepl("guests", description, ignore.case = TRUE) ~ "guests",
                                          grepl("need", description, ignore.case = TRUE) ~ "need",
                                          grepl("stay", description, ignore.case = TRUE) ~ "stay",
                                          grepl("questions", description, ignore.case = TRUE) ~ "questions",
                                          grepl("help", description, ignore.case = TRUE) ~ "help",
                                          grepl("happy", description, ignore.case = TRUE) ~ "happy",
                                          grepl("time", description, ignore.case = TRUE) ~ "time",
                                          grepl("anything", description, ignore.case = TRUE) ~ "anything",
                                          grepl("around", description, ignore.case = TRUE) ~ "around",
                                          grepl("email", description, ignore.case = TRUE) ~ "email",
                                          grepl("please", description, ignore.case = TRUE) ~ "please",
                                          grepl("answer", description, ignore.case = TRUE) ~ "answer",
                                          grepl("love", description, ignore.case = TRUE) ~ "love",
                                          grepl("call", description, ignore.case = TRUE) ~ "call",
                                          grepl("reccomendations", description, ignore.case = TRUE) ~ "reccomendations",
                                          grepl("provide", description, ignore.case = TRUE) ~ "provide",TRUE ~ "other"),
         desc_category = as.factor(interaction_category))

#get rid of original interaction column 
airbnb_test <- airbnb_test%>%
  select(-interaction)

#Column: market

#Looking at market, seems like it groups neighborhoods
#so we can include this in or model instead of neighborhood because market at 56 levels, but neighborhood has 1160 levels 

airbnb_train <- airbnb_train %>% 
  mutate(market = as.factor(market),
         region=as.factor(region))
airbnb_test <- airbnb_test %>% 
  mutate(market = as.factor(market),
         region=as.factor(region))

airbnb_train <- airbnb_train %>% 
  select(-name,-summary,-space,-description,-experiences_offered,-neighborhood_overview,-notes,-transit,-country_code,-access,-house_rules,-host_name,-host_about,-street)

airbnb_test <- airbnb_test %>% 
  select(-name,-summary,-space,-description,-experiences_offered,-neighborhood_overview,-notes,-transit,-country_code,-access,-house_rules,-host_name,-host_about,-street)

# Check for missing values in each variable
missing_values <- colSums(is.na(airbnb_train))

# Print the names of variables with missing values
names_with_missing <- names(missing_values[missing_values > 0])
print(names_with_missing)

# Check for missing values in each variable
missing_values <- colSums(is.na(airbnb_test))
# Print the names of variables with missing values
names_with_missing <- names(missing_values[missing_values > 0])
print(names_with_missing)

airbnb_train <- airbnb_train %>%
  mutate(
    parking = ifelse(is.na(parking), "Not Mentioned", parking),
    pets_allowed = ifelse(is.na(pets_allowed), "Not Mentioned", pets_allowed),
    private_entrance = ifelse(is.na(private_entrance), "Not Mentioned", private_entrance),
    kitchen_access = ifelse(is.na(kitchen_access), "Not Mentioned", kitchen_access),
    smoke_free = ifelse(is.na(smoke_free), "Not Mentioned", smoke_free),
    event_friendly = ifelse(is.na(event_friendly), "Not Mentioned", event_friendly),
    full_access = ifelse(is.na(full_access), "Not Mentioned", full_access)
  ) %>%
  mutate_at(vars(parking, pets_allowed, private_entrance, kitchen_access, smoke_free, event_friendly, full_access), factor)
airbnb_test <- airbnb_test %>%
  mutate(
    parking = ifelse(is.na(parking), "Not Mentioned", parking),
    pets_allowed = ifelse(is.na(pets_allowed), "Not Mentioned", pets_allowed),
    private_entrance = ifelse(is.na(private_entrance), "Not Mentioned", private_entrance),
    kitchen_access = ifelse(is.na(kitchen_access), "Not Mentioned", kitchen_access),
    smoke_free = ifelse(is.na(smoke_free), "Not Mentioned", smoke_free),
    event_friendly = ifelse(is.na(event_friendly), "Not Mentioned", event_friendly),
    full_access = ifelse(is.na(full_access), "Not Mentioned", full_access)
  ) %>%
  mutate_at(vars(parking, pets_allowed, private_entrance, kitchen_access, smoke_free, event_friendly, full_access), factor)


# Columns to convert to factors 
#this has to be done in loop for both test and train this is too long
for (col in names(airbnb_train)) {
  if (startsWith(col, "has_")) {
    airbnb_train[[col]] <- as.factor(airbnb_train[[col]])
  }
}
for (col in names(airbnb_test)) {
  if (startsWith(col, "has_")) {
    airbnb_test[[col]] <- as.factor(airbnb_test[[col]])
  }
}

#---------------EXLORATORY DATA ANALYSIS ON FEATURES------------#
#seeing extra charges relation w cancellation policy
# Count of listings with/without extra charges by cancellation policy
charges_policy_summary <- airbnb_train %>%
  group_by(cancellation_policy, charges_for_extra) %>%
  summarise(Count = n(), .groups = 'drop')

# Plotting
ggplot(charges_policy_summary, aes(x = cancellation_policy, y = Count, fill = charges_for_extra)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Extra Charges by Cancellation Policy",
       x = "Cancellation Policy",
       y = "Count of Listings") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability

#DISTRIBUTION OF INTERACTION CATEGORY VS PERFECT_RATING_SCORE
summary_data <- airbnb_train %>%
  group_by(interaction_category, perfect_rating_score) %>%
  summarise(Count = n(), .groups = 'drop')

# Create the grouped bar chart
ggplot(summary_data, aes(x = interaction_category, y = Count, fill = perfect_rating_score)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_fill_brewer(palette = "Set1") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + # Rotate x labels for better visibility
  labs(title = "Distribution of Interaction Categories by Perfect Rating Score",
       x = "Interaction Category",
       y = "Count",
       fill = "Rating Score")
##PRICE VARIABLE
#hist for price to check variation 
hist(airbnb_train$price)
#doing a log trasformation
hist(log(airbnb_train$price, 10),breaks=10)

#HOUSE RULES

#first_review VS PERFECT_RATING_SCORE
rating_summary <- airbnb_train %>%
  mutate(review_year = format(first_review, "%Y")) %>%
  group_by(review_year, perfect_rating_score) %>%
  summarise(Count = n(), .groups = 'drop')
ggplot(rating_summary, aes(x = review_year, y = Count, fill = perfect_rating_score)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Distribution of Perfect Ratings by Review Year",
       x = "Review Year",
       y = "Count of Ratings",
       fill = "Perfect Rating Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 


#property type vs price
# Calculate the top property types
top_types <- airbnb_train %>%
  group_by(property_type) %>%
  summarise(Count = n(), .groups = 'drop') %>%
  top_n(10, Count) %>%
  pull(property_type)

# Filter and mutate the data
airbnb_train_filtered <- airbnb_train %>%
  mutate(property_type = ifelse(property_type %in% top_types, property_type, "Other"))

# Create the plot with filtered data
ggplot(airbnb_train_filtered, aes(x = property_type, y = price)) +
  geom_boxplot() +
  labs(title = "Price Distribution by Top Property Types and Others",
       x = "Property Type",
       y = "Price") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better visibility

##word cloud
#install.packages("wordcloud")
#install.packages("tm")

# Load packages
library(wordcloud)
library(tm)

table(airbnb_train$host_mentions)

library(RColorBrewer)

# Example data with categories and their frequencies
categories <- c("Culture and Lifestyle", "Education and Learning", "Family and Relationships",
                "Food and Cooking", "Health and Well-being", "Home and Living", "Other",
                "Outdoor and Adventure", "Professional and Business", "Social and Entertainment",
                "Travel and Locations")
frequencies <- c(138, 171, 708, 343, 34, 2551, 30482, 254, 890, 1592, 54904)

# Create a data frame
data <- data.frame(Category = categories, Frequency = frequencies)

# Apply a square root transformation to frequencies to reduce disparity
data$SqrtFrequency = sqrt(data$Frequency)

# Generate the word cloud using the transformed frequencies
# Ensure the plotting window is sufficiently large
png("wordcloud.png", width = 2000, height = 1000)  # Adjust dimensions as needed
wordcloud(words = data$Category, freq = data$SqrtFrequency, min.freq = sqrt(min(data$Frequency)),
          scale = c(4, 1),  # Adjusting scale to fit all words
          colors = brewer.pal(8, "Dark2"),  # Use a color palette from RColorBrewer
          random.order = FALSE,  # Place the most frequent words in the center
          rot.per = 0.25)  # Fraction of words with 90 degree rotation
##LISTING PRICE DISTRIBUTION THROUGH REGIONS



# Get US map data
us_map <- map_data("state")

# Plotting
ggplot(data = airbnb_train, aes(x = longitude, y = latitude)) +
  # Draw the map
  geom_polygon(data = us_map, aes(x = long, y = lat, group = group), fill = "white", color = "black") +
  # Add heatmap layer
  geom_bin2d(aes(fill = ..count..), bins = 100, alpha = 0.5) + # adjust bins for better resolution or performance
  scale_fill_gradient(low = "blue", high = "red", name = "Listing Density") +
  labs(title = "Heatmap of Airbnb Listing Density Across the U.S.",
       x = "Longitude", y = "Latitude") +
  coord_fixed(1.3)  # Maintain aspect ratio


## Longitude and latitude bounds for the San Francisco Bay Area
lon_min <- -123.0
lon_max <- -121.5
lat_min <- 37.0
lat_max <- 38.5

# Get US map data
us_map <- map_data("state")

# Plotting with zoom to a specific region
ggplot(data = airbnb_train, aes(x = longitude, y = latitude)) +
  # Draw the map, focusing on the specified bounds
  geom_polygon(data = us_map, aes(x = long, y = lat, group = group), fill = "white", color = "black") +
  # Add heatmap layer
  geom_bin2d(aes(fill = ..count..), bins = 100, alpha = 0.5) + 
  scale_fill_gradient(low = "blue", high = "red", name = "Listing Density") +
  labs(title = "Heatmap of Airbnb Listing Density in the San Francisco Bay Area",
       x = "Longitude", y = "Latitude") +
  coord_fixed(1.3) +
  xlim(lon_min, lon_max) +  # Set longitude limits
  ylim(lat_min, lat_max)    # Set latitude limits

#PRICE DISTRIBUTION


airbnb_train$log_price <- log1p(airbnb_train$price)
airbnb_test$log_price <- log1p(airbnb_test$price)
sum(is.na(airbnb_train$log_price ))
airbnb_train$log_price
airbnb_train$log_monthly_price <- log1p(airbnb_train$monthly_price)
airbnb_test$log_monthly_price <- log1p(airbnb_test$monthly_price)
airbnb_train$log_weekly_price <- log1p(airbnb_train$weekly_price)
airbnb_test$log_weekly_price <- log1p(airbnb_test$weekly_price)
airbnb_train$log_cleaning_fee <- log1p(airbnb_train$cleaning_fee )
airbnb_test$log_cleaning_fee  <- log1p(airbnb_test$cleaning_fee )

#

## removing some redudant features after running lasso 
# Remove columns
airbnb_train <- airbnb_train %>%
  select(-log_weekly_price, -log_monthly_price, -cleaning_fee)


airbnb_test <- airbnb_test %>% 
  select(-log_weekly_price,-log_monthly_price,-cleaning_fee)

#replacing spaces 
names(airbnb_train) <- gsub(" ", "_", names(airbnb_train))

# View the modified variable names
names(airbnb_train)

names(airbnb_test) <- gsub(" ", "_", names(airbnb_test))

##--TRAIN TEST FEATURES SAME-----#
# Perform feature selection using Lasso regularization

train_features <- colnames(airbnb_train)
test_features <- colnames(airbnb_test)

# Find the common features
common_features <- intersect(train_features, test_features)

# Display the common features
print(common_features)

# Subset full_train_data_X to include only selected features
airbnb_train_X <- airbnb_train[, common_features]
perfect_rating_score <- airbnb_train$perfect_rating_score
airbnb_train <-cbind(airbnb_train_X,perfect_rating_score)
# Subset test_X to include only selected features
test_X <- airbnb_test[, common_features]

##-----checking levels--------##
factor_columns_train <- sapply(airbnb_train, is.factor)
factor_columns_test <- sapply(test_X, is.factor)

# Get the names of factor columns
factor_names_train <- names(airbnb_train)[factor_columns_train]
factor_names_test <- names(test_X)[factor_columns_test]

# Compare factor levels
for (factor_name in intersect(factor_names_train, factor_names_test)) {
  train_levels <- levels(airbnb_train[[factor_name]])
  test_levels <- levels(test_X[[factor_name]])
  
  if (!identical(train_levels, test_levels)) {
    cat("Factor", factor_name, "has different levels in the train and test datasets.\n")
    cat("Train levels:", train_levels, "\n")
    cat("Test levels:", test_levels, "\n")
    
    # Check for additional levels in train or test
    extra_levels_train <- setdiff(train_levels, test_levels)
    extra_levels_test <- setdiff(test_levels, train_levels)
    
    if (length(extra_levels_train) > 0) {
      cat("Extra levels in train:", extra_levels_train, "\n")
    } else if (length(extra_levels_test) > 0) {
      cat("Extra levels in test:", extra_levels_test, "\n")
    }
  }
}

for (factor_name in intersect(factor_names_train, factor_names_test)) {
  train_levels <- levels(airbnb_train[[factor_name]])
  test_levels <- levels(test_X[[factor_name]])
  
  if (!identical(train_levels, test_levels)) {
    cat("Factor", factor_name, "has different levels in the train and test datasets.\n")
  }
}
#___ new datasets____#
# Create a vector of feature names to remove
features_to_remove <- c("host_neighbourhood", "neighborhood", "city", "market",
                        "smart_location", "cancellation_policy", "has_space_features",
                        "jurisdiction_state", "desc_category")

# Remove these features from your dataset
airbnb_train <- airbnb_train[, -which(names(airbnb_train) %in% features_to_remove)]
test_X <- test_X[, -which(names(test_X) %in% features_to_remove)]

#####EXTRA DATASET#############
summary(train_extra)
library(dplyr)

# Assuming df1 has 'zipcode' and df2 has 'zip'
# First, remove duplicates from df1 based on 'zipcode'
train_data_extra_distinct <- train_extra %>% distinct(zip, .keep_all = TRUE)

# Now, perform the join
joined_train_extra <- airbnb_train %>%
  inner_join(train_data_extra_distinct, by = c("zipcode" = "zip"))

# Check the number of rows in the merged dataset
nrow(joined_train_extra)


#---- Data Split for extra dataset ------
set.seed(123)

# Randomly select 30% of rows for validation
valid_instn_extra <- sample(nrow(joined_train_extra), 0.30 * nrow(joined_train_extra))
validation_data_extra <- joined_train_extra[valid_instn_extra, ]
train_data_extra <- joined_train_extra[-valid_instn_extra, ]

# Keep 10% of validation data aside for final validation
valid_instn_1_extra <- sample(nrow(validation_data_extra), 0.10 * nrow(validation_data_extra))
validation_data_final_extra <- validation_data_extra[valid_instn_1_extra, ]
validation_data_use_extra <- validation_data_extra[-valid_instn_1_extra, ]

# Prepare the training and validation datasets
train_X_extra <- train_data_extra %>% select(-perfect_rating_score)
train_y_extra <- train_data_extra$perfect_rating_score
valid_X_extra <- validation_data_use_extra %>% select(-perfect_rating_score)
valid_y_extra <- validation_data_use_extra$perfect_rating_score

rf_model_extra <- ranger(
  dependent.variable.name = "perfect_rating_score",
  x = train_X_extra,
  y=train_y_extra,
  num.trees = 600,
  mtry = sqrt(185),
  importance = 'impurity',
  probability = TRUE,
  class.weights = c(1.5, 3.5) # Adjust these weights as needed
)

# Make predictions on the validation set
valid_preds_probs_extra <- predict(rf_model_extra, data = valid_X_extra, type = "response")$predictions[, 2]

# Adjust the probability threshold
threshold <- 0.45  # Adjust this value as needed

# Convert probabilities to class predictions
valid_preds_rf_extra <- ifelse(valid_preds_probs_extra >= threshold,"YES","NO")
correct_classifications_rf_extra <- ifelse(valid_preds_rf_extra == valid_y_extra, "YES", "NO")
accuracy_rf_extra <- mean(correct_classifications_rf_extra == "YES")  # Calculating accuracy

cat("Accuracy:", accuracy_rf_extra, "\n")

# Create a confusion matrix
conf_mat_rf_extra <- table(Predicted = valid_preds_rf_extra, Actual = valid_y_extra)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR_rf_extra <- conf_mat_rf_extra["YES", "YES"] / sum(conf_mat_rf_extra[, "YES"])

# Calculate False Positive Rate (FPR)
FPR_rf_extra <- conf_mat_rf_extra["YES", "NO"] / sum(conf_mat_rf_extra[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR_rf_extra))
print(paste("False Positive Rate (FPR):", FPR_rf_extra))
print("Confusion Matrix:")
print(conf_mat_rf_extra)


#-------DATA SPLITTING-------------------------#

set.seed(123)

# Randomly select 30% of rows for validation
valid_instn <- sample(nrow(airbnb_train), 0.30 * nrow(airbnb_train))
validation_data <- airbnb_train[valid_instn, ]
train_data <- airbnb_train[-valid_instn, ]

# Keep 10% of validation data aside for final validation
valid_instn_1 <- sample(nrow(validation_data), 0.10 * nrow(validation_data))
validation_data_final <- validation_data[valid_instn_1, ]
validation_data_use <- validation_data[-valid_instn_1, ]


##----------------feature selection through lasso---------------##


# Set seed for reproducibility
set.seed(123)


# Perform feature selection using Lasso regularization

train_data_X <-as.matrix(train_data%>%select(-perfect_rating_score))
train_data_y <-train_data$perfect_rating_score
valid_X <-as.matrix(validation_data_use%>%select(-perfect_rating_score))
valid_y <-validation_data_use$perfect_rating_score

lasso_model <- cv.glmnet(train_data_X,train_data_y, alpha = 1,family = "binomial",n_folds=5)

# Getting the optimal lambda (penalty parameter) selected by cross-validation
optimal_lambda <- lasso_model$lambda.min

# Fit Lasso model with the optimal lambda
lasso_fit <- glmnet(train_data_X,train_data_y, alpha = 1,family = "binomial", lambda = optimal_lambda)

# Get coefficients and select non-zero ones
lasso_coef <- coef(lasso_fit)
non_zero_indices <- which(lasso_coef != 0)  # Indices of non-zero coefficients
lasso_coef <- as.matrix(lasso_coef[non_zero_indices, , drop = FALSE])  # Drop = FALSE to keep matrix structure

# Get the names of selected features (excluding intercept if present)
selected_feature_names <- rownames(lasso_coef)[-1]  # Assuming the first row is the intercept

# Subset the original dataset to retain the data types
selected_features <- train_data %>% 
  select(all_of(selected_feature_names))

# Show the structure of the selected features to confirm data types
str(selected_features)

##SEEING TRAINING ACCURACY
#predicting through lasso
prediction_lasso_training <- predict(lasso_fit, newx = train_data_X, type = "response")
classification_lasso_training <- ifelse(prediction_lasso_training > 0.5, "YES", "NO")
correct_classifications_lasso_t <- ifelse(classification_lasso_training == train_data_y, "YES", "NO")
accuracy_t<- mean(correct_classifications_lasso_t == "YES")  # Calculating accuracy

cat("Accuracy_Training:", accuracy_t, "\n")

# Create a confusion matrix
conf_mat_training <- table(Predicted = classification_lasso_training, Actual = train_data_y)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR_t <- conf_mat_training["YES", "YES"] / sum(conf_mat_training[, "YES"])

# Calculate False Positive Rate (FPR)
FPR_t<- conf_mat_training["YES", "NO"] / sum(conf_mat_training[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR_t))
print(paste("False Positive Rate (FPR):", FPR_t))
print("Confusion Matrix:")
print(conf_mat_training)

#predicting through lasso
prediction_lasso <- predict(lasso_fit, newx = valid_X, type = "response")
classification_lasso <- ifelse(prediction_lasso > 0.5, "YES", "NO")
correct_classifications_lasso <- ifelse(classification_lasso == valid_y, "YES", "NO")
accuracy <- mean(correct_classifications_lasso == "YES")  # Calculating accuracy

cat("Accuracy_Validation:", accuracy, "\n")

# Create a confusion matrix
conf_mat <- table(Predicted = classification_lasso, Actual = valid_y)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR <- conf_mat["YES", "YES"] / sum(conf_mat[, "YES"])

# Calculate False Positive Rate (FPR)
FPR <- conf_mat["YES", "NO"] / sum(conf_mat[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR))
print(paste("False Positive Rate (FPR):", FPR))
print("Confusion Matrix:")
print(conf_mat)



###-----------MODEL BUILDING------------##
set.seed(123)
##RANDOM FOREST

# Prepare the training and validation datasets
train_X <- train_data %>% select(-perfect_rating_score)
train_y <- train_data$perfect_rating_score
valid_X <- validation_data_use %>% select(-perfect_rating_score)
valid_y <- validation_data_use$perfect_rating_score

rf_model <- ranger(
  dependent.variable.name = "perfect_rating_score",
  x = train_X,
  y=train_y,
  num.trees = 600,
  mtry = sqrt(185),
  importance = 'impurity',
  probability = TRUE,
  class.weights = c(1.5, 3.5) # Adjust these weights as needed
)
##SEEING ON TRAINING DATA FIRST
# Make predictions on the validation set
train_preds_probs <- predict(rf_model, data = train_X, type = "response")$predictions[, 2]

# Adjust the probability threshold
threshold <- 0.46  # Adjust this value as needed

# Convert probabilities to class predictions
train_preds_rf <- ifelse(train_preds_probs >= threshold,"YES","NO")
train_correct_classifications_rf <- ifelse(train_preds_rf == train_y, "YES", "NO")
accuracy_train <- mean(train_correct_classifications_rf == "YES")  # Calculating accuracy

cat("Accuracy_training:", accuracy_train, "\n")

# Create a confusion matrix
conf_mat_rf_train <- table(Predicted = train_preds_rf, Actual = train_y)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR_rf_t <- conf_mat_rf_train["YES", "YES"] / sum(conf_mat_rf_train[, "YES"])

# Calculate False Positive Rate (FPR)
FPR_rf_t <- conf_mat_rf_train["YES", "NO"] / sum(conf_mat_rf_train[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR_rf_t))
print(paste("False Positive Rate (FPR):", FPR_rf_t))
print("Confusion Matrix:")
print(conf_mat_rf_train)

##ON VALIDATION

# Make predictions on the validation set
valid_preds_probs <- predict(rf_model, data = valid_X, type = "response")$predictions[, 2]

# Adjust the probability threshold
threshold <- 0.46  # Adjust this value as needed

# Convert probabilities to class predictions
valid_preds_rf <- ifelse(valid_preds_probs >= threshold,"YES","NO")
correct_classifications_rf <- ifelse(valid_preds_rf == valid_y, "YES", "NO")
accuracy_valid_rf <- mean(correct_classifications_rf == "YES")  # Calculating accuracy

cat("Accuracy:", accuracy_valid_rf , "\n")

# Create a confusion matrix
conf_mat_rf <- table(Predicted = valid_preds_rf, Actual = valid_y)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR_rf <- conf_mat_rf["YES", "YES"] / sum(conf_mat_rf[, "YES"])

# Calculate False Positive Rate (FPR)
FPR_rf <- conf_mat_rf["YES", "NO"] / sum(conf_mat_rf[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR_rf))
print(paste("False Positive Rate (FPR):", FPR_rf))
print("Confusion Matrix:")
print(conf_mat_rf)

# Get variable importance
importance_scores <- rf_model$variable.importance
# Convert importance scores to a data frame
importance_df <- data.frame(Feature = names(importance_scores), Importance = importance_scores)

# Order by importance and select the top 10
top_10_features <- importance_df %>%
  dplyr::arrange(desc(Importance)) %>%
  dplyr::slice(1:10)

# Plotting the top 10 important features
ggplot(top_10_features, aes(x = reorder(Feature, Importance), y = Importance, fill = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flip coordinates for horizontal bars
  scale_fill_gradient(low = "skyblue", high = "salmon") +
  labs(title = "Top 10 Important Features in Random Forest Model",
       x = "Importance",
       y = "Features") +
  theme_minimal()

library(ranger)

set.seed(123)

# Initialize vectors to store the results
train_sizes <- seq(0.1, 1, by = 0.1) # Training sizes from 10% to 100%
accuracy <- numeric(length(train_sizes))
TPR <- numeric(length(train_sizes))
FPR <- numeric(length(train_sizes))

# Loop over different sizes of the training set
for (i in seq_along(train_sizes)) {
  # Calculate the number of data points to use
  train_size <- floor(train_sizes[i] * nrow(train_data))
  
  # Sample from the training data
  train_subset <- train_data[sample(nrow(train_data), train_size), ]
  
  # Prepare the training and validation datasets
  train_X_subset <- train_subset %>% select(-perfect_rating_score)
  train_y_subset <- train_subset$perfect_rating_score
  
  # Train the model on the subset
  rf_model_subset <- ranger(
    dependent.variable.name = "perfect_rating_score",
    data = train_subset,
    num.trees = 600,
    mtry = sqrt(185),
    importance = 'impurity',
    probability = TRUE,
    class.weights = c(1.5, 3.5) # Adjust these weights as needed
  )
  
  # Make predictions on the validation set
  valid_preds_probs <- predict(rf_model_subset, data = valid_X, type = "response")$predictions[, 2]
  
  # Adjust the probability threshold
  threshold <- 0.46  # Adjust this value as needed
  
  # Convert probabilities to class predictions
  valid_preds_rf <- ifelse(valid_preds_probs >= threshold, "YES", "NO")
  
  # Calculate accuracy
  accuracy[i] <- mean(valid_preds_rf == valid_y)
  
  # Create a confusion matrix
  conf_mat <- table(Predicted = valid_preds_rf, Actual = valid_y)
  
  # Calculate True Positive Rate (TPR) or Sensitivity
  TPR[i] <- conf_mat["YES", "YES"] / sum(conf_mat[, "YES"])
  
  # Calculate False Positive Rate (FPR)
  FPR[i] <- conf_mat["YES", "NO"] / sum(conf_mat[, "NO"])
}

# Plot the learning curve
plot(train_sizes, accuracy, type = "b", pch = 19, col = "blue",
     xlab = "Training Set Size", ylab = "Accuracy",
     main = "Learning Curve for Random Forest Model")

# Create a data frame to store the results
results_df <- data.frame(
  Train_Size = train_sizes,
  Accuracy = accuracy,
  TPR = TPR,
  FPR = FPR
)

# Plot the fitting curve
ggplot(results_df, aes(x = Train_Size, y = Accuracy)) +
  geom_line(color = "blue") +
  labs(title = "Fitting Curve: Accuracy vs. Training Set Size",
       x = "Training Set Size", y = "Accuracy")

# Plot the ROC curve
ggplot(results_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "red") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = "ROC Curve", x = "False Positive Rate (FPR)", y = "True Positive Rate (TPR)")


#######boosting ###############
##XGBOOST
#Filter out the (Intercept) term if it exists
train_data_selected <- train_data %>% 
  select(all_of(c(selected_feature_names, "perfect_rating_score")))

validation_data_selected <- validation_data_use %>% 
  select(all_of(c(selected_feature_names, "perfect_rating_score")))

# Separate the target variable first
train_X <- train_data_selected %>% select(-perfect_rating_score)
train_y <- train_data_selected$perfect_rating_score
valid_X <- validation_data_selected %>% select(-perfect_rating_score)
valid_y <- validation_data_selected$perfect_rating_score

# Specify XGBoost parameters
numeric_target <- as.integer(train_data_selected$perfect_rating_score == "YES")

# Check the range of the encoded labels
label_range <- range(numeric_target)
print(label_range)

bst_train_y<-train_y
bst_valid_y <- valid_y

# Remove the target variable from the datasets
bst_train_features <- train_data_selected %>% select(-perfect_rating_score)
bst_valid_features <- validation_data_selected %>% select(-perfect_rating_score)

# Create dummy variables only on the feature sets
dummies <- dummyVars(" ~ .", data = bst_train_features)
transformed_train_data_X <- predict(dummies, newdata = bst_train_features)
transformed_valid_data_X <- predict(dummies, newdata = bst_valid_features)

# Convert the data frames to matrix format as needed for xgboost
bst_train_data_X <- as.matrix(transformed_train_data_X)
bst_valid_X <- as.matrix(transformed_valid_data_X)

# Now you can proceed with xgboost model preparation
dtrain <- xgb.DMatrix(data = bst_train_data_X, label = numeric_target)
dvalid <- xgb.DMatrix(data = bst_valid_X, label=bst_valid_y)

# Train the XGBoost model
xgb_model <- xgboost(
  data = dtrain,
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 47,  # Reduced depth
  eta = 0.18,  # Slightly reduced learning rate
  nthread = 3,
  gamma = 5.9,  # Increased gamma
  scale_pos_weight = 0.72,  # Adjust based on the ratio of negative to positive cases
  min_child_weight = 3,  # Could increase if necessary
  nrounds = 50  # Increased rounds to accommodate lower learning rate
)
#Evaluate model on training set
# Evaluate the model
bst_pred_train <- predict(xgb_model, dtrain)
# Adjust the probability threshold
threshold <- 0.45  # Adjust this value as needed

train_data_selected$perfect_rating_score <- ifelse(train_data_selected$perfect_rating_score == 1, "YES", "NO")
bst_train_data_y <- train_data_selected$perfect_rating_score
bst_valid_y <- validation_data_use$perfect_rating_score

# Convert probabilities to class predictions
classification_bst_train <- ifelse(bst_pred_train >= threshold, "YES", "NO")
correct_classifications_bst_train <- ifelse(classification_bst_train == bst_train_y, "YES", "NO")
accuracy_train_xg <- mean(correct_classifications_bst_train == "YES")  # Calculating accuracy

cat("Training Accuracy:", accuracy_train_xg, "\n")

# Create a confusion matrix
conf_mat_xgboost_train <- table(Predicted = classification_bst_train, Actual = bst_train_y)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR_xgboost_train <- conf_mat_xgboost_train["YES", "YES"] / sum(conf_mat_xgboost_train[, "YES"])

# Calculate False Positive Rate (FPR)
FPR_xgboost_train  <- conf_mat_xgboost_train["YES", "NO"] / sum(conf_mat_xgboost_train[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR_xgboost_train ))
print(paste("False Positive Rate (FPR):", FPR_xgboost_train ))
print("Confusion Matrix:")
print(conf_mat_xgboost_train)

# Evaluate the modelon validation data 
bst_pred <- predict(xgb_model, dvalid)
# Adjust the probability threshold
threshold <- 0.45  # Adjust this value as needed

train_data_selected$perfect_rating_score <- ifelse(train_data_selected$perfect_rating_score == 1, "YES", "NO")
bst_train_data_y <- train_data_selected$perfect_rating_score
bst_valid_y <- validation_data_use$perfect_rating_score
# Convert probabilities to class predictions
classification_bst <- ifelse(bst_pred >= threshold, "YES", "NO")
correct_classifications_bst <- ifelse(classification_bst == bst_valid_y, "YES", "NO")
accuracy <- mean(correct_classifications_bst == "YES")  # Calculating accuracy

cat("Accuracy:", accuracy, "\n")

# Create a confusion matrix
conf_mat <- table(Predicted = classification_bst, Actual = bst_valid_y)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR <- conf_mat["YES", "YES"] / sum(conf_mat[, "YES"])

# Calculate False Positive Rate (FPR)
FPR <- conf_mat["YES", "NO"] / sum(conf_mat[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR))
print(paste("False Positive Rate (FPR):", FPR))
print("Confusion Matrix:")
print(conf_mat)

##LIGHTGBM MODEL
numeric_target <- as.integer(train_data$perfect_rating_score == "YES")
train_X <- train_data %>% select(-perfect_rating_score)
train_y <- train_data$perfect_rating_score
valid_X <- validation_data_use %>% select(-perfect_rating_score)
valid_y <- validation_data_use$perfect_rating_score

train_X_matrix<- as.matrix(train_X)
valid_X_matrix <- as.matrix(valid_X)
valid_y <- validation_data_use$perfect_rating_score



# Define parameters for the LightGBM model
params <- list(
  objective = "binary",                    # Objective function
  metric = "binary_error",                # Performance evaluation metric
  learning_rate = 0.28,                     # Learning rate
  num_leaves = 55,                         # Maximum number of leaves in one tree
  max_depth = 34,                          # Maximum tree depth for base learners, -1 means no limit
  min_data_in_leaf = 35,                   # Minimum number of data in one leaf
  bagging_fraction = 0.8,                 # Fraction of data to be used for each iteration
  bagging_freq =2,                        # Frequency of bagging
  feature_fraction = 0.8,                # Fraction of features to be used for each iteration
  verbosity = 1                           # Level of LightGBM's verbosity
)

# Calculate the class weights for unbalanced classes
params$scale_pos_weight <- 0.9 # Adjusting scale_pos_weight

# Re-train the model with the new parameter
lgb_model <- lgb.train(
  params,
  lgb.Dataset(data = train_X_matrix, label = numeric_target),
  valids = list(valid = lgb.Dataset(data = valid_X_matrix, label = valid_y)),
  nrounds = 200,                  # Number of boosting rounds
  early_stopping_rounds = 15      # Early stopping if no improvement for 15 rounds
)
##ON TRAINING DATA
lightGBMpredictions_train <- predict(lgb_model, train_X_matrix)

# Convert predicted probabilities to binary predictions
lightGBM_prediction_train <- ifelse(lightGBMpredictions_train > 0.5,"YES","NO")
# Convert probabilities to class predictions
correct_classifications_GBM_train <- ifelse(lightGBM_prediction_train == train_y, "YES", "NO")
accuracy_lgbm_train <- mean(correct_classifications_GBM_train == "YES")  # Calculating accuracy

cat("Accuracy_Training:", accuracy_lgbm_train, "\n")

# Create a confusion matrix
conf_mat_lgbm_train <- table(Predicted = lightGBM_prediction_train, Actual = train_y)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR_lgbm_train <- conf_mat_lgbm_train["YES", "YES"] / sum(conf_mat_lgbm_train[, "YES"])

# Calculate False Positive Rate (FPR)
FPR_lgbm_train <- conf_mat_lgbm_train["YES", "NO"] / sum(conf_mat_lgbm_train[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR_lgbm_train))
print(paste("False Positive Rate (FPR):", FPR_lgbm_train))
print("Confusion Matrix:")
print(conf_mat_lgbm_train)


#ON VALIDATION DATA
# Make predictions on the validation data
lightGBMpredictions <- predict(lgb_model, valid_X_matrix)



# Convert predicted probabilities to binary predictions
lightGBM_predictions <- ifelse(lightGBMpredictions > 0.5,"YES","NO")
# Convert probabilities to class predictions
correct_classifications_GBM <- ifelse(lightGBM_predictions == valid_y, "YES", "NO")
accuracy <- mean(correct_classifications_GBM == "YES")  # Calculating accuracy

cat("Accuracy:", accuracy, "\n")

# Create a confusion matrix
conf_mat <- table(Predicted = lightGBM_predictions, Actual = valid_y)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR <- conf_mat["YES", "YES"] / sum(conf_mat[, "YES"])

# Calculate False Positive Rate (FPR)
FPR <- conf_mat["YES", "NO"] / sum(conf_mat[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR))
print(paste("False Positive Rate (FPR):", FPR))
print("Confusion Matrix:")
print(conf_mat)

#KNN


#KNN
####### knn #############

train_X <- train_data %>% select(-perfect_rating_score)
train_y <- train_data$perfect_rating_score
valid_X <- validation_data_use %>% select(-perfect_rating_score)
valid_y <- validation_data_use$perfect_rating_score

train_X_matrix<- as.matrix(train_X)
valid_X_matrix <- as.matrix(valid_X)
valid_y <- validation_data_use$perfect_rating_score

#these are character matric, have to make them numeric
numeric_train_X <- train_X %>%
  mutate_if(is.factor, as.numeric) %>%
  mutate_if(is.character, function(x) as.numeric(as.factor(x)))

numeric_valid_X <- valid_X %>%
  mutate_if(is.factor, as.numeric) %>%
  mutate_if(is.character, function(x) as.numeric(as.factor(x)))

reference_date <- as.Date("2000-01-01")  # Choose a suitable reference date
numeric_train_X <- numeric_train_X %>%
  mutate_if(is.Date, function(x) as.numeric(x - reference_date))

numeric_valid_X <- numeric_valid_X %>%
  mutate_if(is.Date, function(x) as.numeric(x - reference_date))

num_train_X_matrix <- as.matrix(numeric_train_X)
num_valid_X_matrix <- as.matrix(numeric_valid_X)

# Choose the number of neighbors
k <- 13  
# Train the KNN model and make predictions on the train set
knn_predictions_train <- knn(train = num_train_X_matrix, test = num_train_X_matrix, cl = train_y, k = k)

# Calculate accuracy
knn_accuracy_train <- sum(knn_predictions == train_y) / length(train_y)
cat("KNN Model Accuracy_train:", knn_accuracy_train, "\n")

# Create a confusion matrix
knn_conf_mat_train<- table(Predicted = knn_predictions_train, Actual = train_y)

# Print the confusion matrix
print(knn_conf_mat_train)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR_knn_t <- knn_conf_mat_train["YES", "YES"] / sum(knn_conf_mat_train[, "YES"])

# Calculate False Positive Rate (FPR)
FPR_knn_t <- knn_conf_mat_train["YES", "NO"] / sum(knn_conf_mat_train[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR_knn_t))
print(paste("False Positive Rate (FPR):", FPR_knn_t))


# Train the KNN model and make predictions on the validation set
knn_predictions <- knn(train = num_train_X_matrix, test = num_valid_X_matrix, cl = train_y, k = k)

# Calculate accuracy
knn_accuracy <- sum(knn_predictions == valid_y) / length(valid_y)
cat("KNN Model Accuracy:", knn_accuracy, "\n")

# Create a confusion matrix
knn_conf_mat <- table(Predicted = knn_predictions, Actual = valid_y)

# Print the confusion matrix
print(knn_conf_mat)
# Calculate True Positive Rate (TPR) or Sensitivity
TPR_knn <- knn_conf_mat["YES", "YES"] / sum(knn_conf_mat[, "YES"])

# Calculate False Positive Rate (FPR)
FPR_knn <- knn_conf_mat["YES", "NO"] / sum(knn_conf_mat[, "NO"])

# Print TPR, FPR, and confusion matrix
# Print TPR, FPR, and confusion matrix
print(paste("True Positive Rate (TPR):", TPR_knn))
print(paste("False Positive Rate (FPR):", FPR_knn))

##LOGISTIC REGRESSION

k <-15
selected_features_glm <- selected_feature_names[selected_feature_names != "(Intercept)"]
logistic_formula_str <- paste("perfect_rating_score ~", paste(selected_features_glm, collapse = "+"))
logistic_formula <- as.formula(logistic_formula_str)
logistic_formula

# Initialize vectors to store TPR and FPR
tpr_values <- numeric(k)
fpr_values <- numeric(k)
accuracy_values <- numeric(k)
##TRAIN SET
# Perform k-fold cross-validation
for (i in 1:k) {
  # Split data into training and validation sets
  folds <- cut(seq(1, nrow(airbnb_train)), breaks = k, labels = FALSE)
  validation_indices <- which(folds == i, arr.ind = TRUE)
  validation_set <- airbnb_train[validation_indices, ]
  training_set <- airbnb_train[-validation_indices, ]
  
  # model
  model <- glm(logistic_formula, data = training_set, family = binomial)
  
  # prediction
  prediction_train <- predict(model, newdata = training_set, type = "response")
  
  #classifications 
  predicted_classes_t <- factor(ifelse(prediction_train > 0.5, "YES", "NO"), levels = levels(training_set$perfect_rating_score))
  
  # cm
  confusion_matrix_t <- table(training_set$perfect_rating_score, predicted_classes_t)
  
  # tpr and fpr 
  tpr_values[i] <- confusion_matrix_t[2, 2] / sum(confusion_matrix_t[2, ])
  fpr_values[i] <- confusion_matrix_t[1, 2] / sum(confusion_matrix_t[1, ])
  accuracy_values[i] <- sum(diag(confusion_matrix_t)) / sum(confusion_matrix_t)
  
}


#average tpr and fpr 
mean_tpr <- mean(tpr_values)
mean_fpr <- mean(fpr_values)
cat("Mean TPR:", mean_tpr, "\n")
cat("Mean FPR:", mean_fpr, "\n")
tpr_values
fpr_values
mean_accuracy <- mean(accuracy_values)
mean_accuracy



##VALIDATION SET
# Perform k-fold cross-validation
for (i in 1:k) {
  # Split data into training and validation sets
  folds <- cut(seq(1, nrow(airbnb_train)), breaks = k, labels = FALSE)
  validation_indices <- which(folds == i, arr.ind = TRUE)
  validation_set <- airbnb_train[validation_indices, ]
  training_set <- airbnb_train[-validation_indices, ]
  
  # model
  model <- glm(logistic_formula, data = training_set, family = binomial)
  
  # prediction
  predictions <- predict(model, newdata = validation_set, type = "response")
  
  #classifications 
  predicted_classes <- factor(ifelse(predictions > 0.5, "YES", "NO"), levels = levels(validation_set$perfect_rating_score))
  
  # cm
  confusion_matrix <- table(validation_set$perfect_rating_score, predicted_classes)
  
  # tpr and fpr 
  tpr_values[i] <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
  fpr_values[i] <- confusion_matrix[1, 2] / sum(confusion_matrix[1, ])
  accuracy_values[i] <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
}


#average tpr and fpr 
mean_tpr <- mean(tpr_values)
mean_fpr <- mean(fpr_values)
cat("Mean TPR:", mean_tpr, "\n")
cat("Mean FPR:", mean_fpr, "\n")
tpr_values
fpr_values
mean_accuracy <- mean(accuracy_values)
mean_accuracy





##TESTING DATA
##TESTING DATA
## we have different features/columns in train and test so to make code run i selected common features/columns and
#subsetted those as the new datasets, then the whole airbnb_train is used , and predicting on test
#above we didnt have to do the feature subsetting because we were prediciting on validation
#and validation and train both came from airbnb_train dataset so they had same columns

set.seed(123)

airbnb_train1 <- airbnb_train %>% 
  select(-perfect_rating_score)

full_train_data_X_selected <-as.matrix(airbnb_train1)
full_train_data_y <-airbnb_train$perfect_rating_score
test_X_matrix <-as.matrix(test_X)

lasso_model_test <- cv.glmnet(full_train_data_X_selected,full_train_data_y, alpha = 1,family = "binomial",n_folds=5)

# Get the optimal lambda (penalty parameter) selected by cross-validation
optimal_lambda_test <- lasso_model_test$lambda.min

# Fit Lasso model with the optimal lambda
lasso_fit_test <- glmnet(full_train_data_X_selected,full_train_data_y, alpha = 1,family = "binomial", lambda = optimal_lambda_test)

# Get coefficients and select non-zero ones
lasso_coef_test <- coef(lasso_fit_test)
non_zero_indices_t <- which(lasso_coef_test != 0)  # Indices of non-zero coefficients
lasso_coef_test <- as.matrix(lasso_coef_test[non_zero_indices_t, , drop = FALSE])  # Drop = FALSE to keep matrix structure

# Get the names of selected features (excluding intercept if present)
selected_feature_names_t <- rownames(lasso_coef_test)[-1]  # Assuming the first row is the intercept

# Subset the original dataset to retain the data types
selected_features_test <- airbnb_train %>% 
  select(all_of(selected_feature_names_t))

# Show the structure of the selected features to confirm data types
str(selected_features_test)

print("Selected Features:")
print(selected_features_test)

# Make predictions on the subset of test data
predictions_lasso_test <- predict(lasso_fit_test, newx = test_X_matrix, type = "response")
classification_lasso_test <- ifelse(predictions_lasso_test > 0.5, "YES", "NO")
classified_df <- data.frame(classification_lasso_test)

write.csv(classified_df, file = "classifications_lasso.csv", row.names = FALSE)

#########random forest 

full_train_data_selected <- airbnb_train1

test_X_data_selected <- test_X

full_train_data_y <-airbnb_train$perfect_rating_score

rf_model_fit <- ranger(
  dependent.variable.name = "perfect_rating_score",
  x = full_train_data_selected,
  y=full_train_data_y,
  num.trees = 600,
  mtry = sqrt(185),
  importance = 'impurity',
  probability = TRUE,
  class.weights = c(1.5, 3.5)# Adjust these weights as needed
)



# Predict probabilities on the test set using the trained ranger model
predictions_rf_test <- predict(rf_model_fit, data = test_X_data_selected, type = "response")

# Extract the probabilities of the positive class (assuming it's the second column)
test_preds_probs <- predictions_rf_test$predictions[, 2]
# Set a threshold for classification
threshold <- 0.46  # Adjust this value as needed

# Convert probabilities to binary predictions
test_preds <- ifelse(test_preds_probs >= threshold, "YES", "NO")
predictions_df <- data.frame(x = test_preds)

write.csv(predictions_df, file = "classifications_rf3.csv", row.names = FALSE)



##########BOOSTING TEST##########

set.seed(123)
library(xgboost)


full_train_data_selected <- airbnb_train1 %>% 
  select(all_of(c(selected_feature_names_t)))

test_X_data_selected <- test_X %>% 
  select(all_of(c(selected_feature_names_t)))

numeric_train_data_y <-as.numeric(airbnb_train$perfect_rating_score=="YES")


# Check the range of the encoded labels
label_range <- range(numeric_train_data_y)
print(label_range)


# Create dummy variables only on the feature sets
dummies <- dummyVars(" ~ .", data = full_train_data_selected)
transformed_train_data_X <- predict(dummies, newdata = full_train_data_selected)
transformed_test_data_X <- predict(dummies, newdata = test_X_data_selected)

# Convert the data frames to matrix format as needed for xgboost
bst_train_data_X <- as.matrix(transformed_train_data_X)
bst_test_X <- as.matrix(transformed_test_data_X)

# Now you can proceed with xgboost model preparation
dtrain_test <- xgb.DMatrix(data = bst_train_data_X, label = numeric_train_data_y)
dtest <- xgb.DMatrix(data = bst_test_X)
# Train the XGBoost model
xgb_model_test <- xgboost(
  data = dtrain_test,
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 47,  # Reduced depth
  eta = 0.18,  # Slightly reduced learning rate
  nthread = 3,
  gamma = 5.9,  # Increased gamma
  scale_pos_weight = 0.72,  # Adjust based on the ratio of negative to positive cases
  min_child_weight = 3.29,  # Could increase if necessary
  nrounds = 100  # Increased rounds to accommodate lower learning rate
)

predictions_xgb_test <- predict(xgb_model, dtest)

# Convert predictions to binary classification
classification_xgb_test1 <- ifelse(predictions_xgb_test > 0.45, "YES", "NO")



write.csv(classification_xgb_test1,"boost_test2.csv", row.names = FALSE)





##########Testing for LightGBM##########

full_train_data_selected <- airbnb_train1 

test_X_data_selected <- test_X
train_data_y <-airbnb_train$perfect_rating_score

train_X_test_matrix<- as.matrix(full_train_data_selected)
test_X_matrix <- as.matrix(test_X_data_selected)
numeric_train_data_y <-as.numeric(airbnb_train$perfect_rating_score=="YES")


# Define parameters for the LightGBM model
params <- list(
  objective = "binary",                    # Objective function
  metric = "binary_error"                 # Performance evaluation metric
  # learning_rate = 0.31,                     # Learning rate
  #num_leaves = 55,                         # Maximum number of leaves in one tree
  #max_depth = 34,                          # Maximum tree depth for base learners, -1 means no limit
  #min_data_in_leaf = 35,                   # Minimum number of data in one leaf
  # bagging_fraction = 0.8,                 # Fraction of data to be used for each iteration
  #bagging_freq =2,                        # Frequency of bagging
  # feature_fraction = 0.8,                # Fraction of features to be used for each iteration
  # verbosity = 1                           # Level of LightGBM's verbosity
)

# Calculate the class weights for unbalanced classes
#params$scale_pos_weight <- 0.77 # Adjusting scale_pos_weight

# Re-train the model with the new parameter
lgb_model_test <- lgb.train(
  params,
  lgb.Dataset(data = train_X_test_matrix, label =numeric_train_data_y ),
  valids = list(valid = lgb.Dataset(data = test_X_matrix)),
  nrounds = 200,                  # Number of boosting rounds
  early_stopping_rounds = 15      # Early stopping if no improvement for 15 rounds
)

# Make predictions on the validation data
lightGBMpredictions <- predict(lgb_model_test, test_X_matrix)



# Convert predicted probabilities to binary predictions
lightGBM_predictions_test <- ifelse(lightGBMpredictions > 0.5,"YES","NO")
write.csv(lightGBM_predictions_test,"light_test1.csv", row.names = FALSE)


