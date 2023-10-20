from types import NoneType
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException 
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Navigates until it reaches potential advertisement
def navigate(the_food):
    # Driver Information
    global driver
    driver_path = os.getenv("DRIVER_PATH")
    driver = webdriver.Chrome(driver_path)
    driver.get("https://www.nutritionvalue.org")

    # Type in food and search
    search_bar = driver.find_element(By.ID, "food_query")
    search_bar.clear()
    search_bar.send_keys(the_food)
    
    # Submit search
    search_btn = driver.find_element(By.CSS_SELECTOR, 'input[type="submit"]')
    search_btn.click()

    # If the food text is exactly the same as that on the website, then it will go straight to the data without a table to select similiar foods from if it's not exact, so try/except for a NoSuchElementException will be thrown to handle the lack of a table
    try:
        food_btn = driver.find_element(By.CLASS_NAME, "table_item_name")
        food_btn.click()

        # Let advertisement load
        time.sleep(2.5)
        driver.back()

        # Create new food button instance to prevent StaleElementReferenceException
        food_btn = driver.find_element(By.CLASS_NAME, "table_item_name")
        food_btn.click()
    except NoSuchElementException:
        pass


# Get the total amount of each nutrient
def get_nutrient_data():
    
    # This path is shared for all nutrients
    beginning_path = '//*[@id="main"]/tbody/tr[4]/td/table/tbody/tr[7]/'

    # For efficiency, the keys are associated with their respective paths until they are iterated, replacing them with the total amount of that nutrient.
    nutrients = {
        'Carbohydrate': "td[3]/table[1]/tbody/",
        'Fiber': "td[3]/table[1]/tbody/",
        'Fat': "td[3]/table[2]/tbody/",
        'Polyunsaturated fatty acids': "td[3]/table[2]/tbody/",
        'Monounsaturated fatty acids': "td[3]/table[2]/tbody/",
        'Sugars': "td[3]/table[1]/tbody/",
        'Sodium, Na': "td[1]/table[2]/tbody/",
        'Saturated fatty acids': "td[3]/table[2]/tbody/",
        'Cholesterol': "td[3]/table[3]/tbody/",
        'Protein': "td[1]/table[3]/tbody/",
        'Vitamin A, RAE': "td[1]/table[1]/tbody/",
        'Vitamin B12': "td[1]/table[1]/tbody/",
        'Vitamin B6': "td[1]/table[1]/tbody/",
        'Vitamin C': "td[1]/table[1]/tbody/",
        'Vitamin D': "td[1]/table[1]/tbody/",
        'Vitamin E (alpha-tocopherol)': "td[1]/table[1]/tbody/",
        'Vitamin K': "td[1]/table[1]/tbody/",
        'Riboflavin': "td[1]/table[1]/tbody/",
        'Niacin': "td[1]/table[1]/tbody/",
        'Folate, DFE': "td[1]/table[1]/tbody/",
        'Thiamin': "td[1]/table[1]/tbody/",
        'Calcium, Ca': "td[1]/table[2]/tbody/",
        'Copper, Cu': "td[1]/table[2]/tbody/",
        'Iron, Fe': "td[1]/table[2]/tbody/",
        'Magnesium, Mg': "td[1]/table[2]/tbody/",
        'Phosphorus, P': "td[1]/table[2]/tbody/",
        'Potassium, K': "td[1]/table[2]/tbody/",
        'Selenium, Se': "td[1]/table[2]/tbody/",
        'Zinc, Zn': "td[1]/table[2]/tbody/"
    }

    # For each nutrient, get its value
    for nutrient_type, ending_path in nutrients.items():
        next_break = True
        num = 0
        word = NoneType
        prev_word = NoneType
        # This loop checks if the word from a path is the same as the specified nutrient
        while word != nutrient_type:
            prev_word = word
            num+=1
            # This system of try/except blocks accounts for disimiliar paths that will occur as the loop continues
            try:
                try:
                    word = driver.find_element(By.XPATH, f'{beginning_path}{ending_path}tr[{num}]/td[1]/a').text.strip()
                except NoSuchElementException:
                    word = driver.find_element(By.XPATH, f'{beginning_path}{ending_path}tr[{num}]/td[1]').text.strip()
            except NoSuchElementException:
                word = NoneType
            if isinstance(word, str) or num > 50: # Have 'num > 50' condition to break out of infinite loop, which occurs if the nutrient is not found
                if isinstance(prev_word, str) or num > 50:
                    if word == prev_word or num > 50:
                        next_break = False
                        break
        
        # Using the num incremented upon by the previous loop, this set of statements takes the path and finds the exact value
        total_nutrients = 0.0
        if next_break:
            nutrient_path = driver.find_element(By.XPATH, f'{beginning_path}{ending_path}tr[{num}]/td[2]')
            nutrient_string = nutrient_path.text
            if nutrient_string != "":
                nutrient_inc = 0
                nutrient_text = ""
                while nutrient_string[nutrient_inc].isnumeric() or nutrient_string[nutrient_inc] == '.':
                    nutrient_text += nutrient_string[nutrient_inc]
                    nutrient_inc += 1
                total_nutrients = float(nutrient_text)
        
        nutrients.__setitem__(nutrient_type, total_nutrients)

    return nutrients


# Calculate the rank using the nutrient data within the formula
def calculate_rank(nutrients):
    #Unsaturated fat
    unsaturated_fat = nutrients.get('Polyunsaturated fatty acids') + nutrients.get('Monounsaturated fatty acids')
    #Trans
    trans_fat = nutrients.get('Fat') - unsaturated_fat - nutrients.get('Saturated fatty acids')
    #Starch
    starch = nutrients.get('Carbohydrate') - nutrients.get('Sugars') - nutrients.get('Fiber')
    #Calories
    calories = float(driver.find_element(By.XPATH, '//*[@id="calories"]').text)

    # Ratios for scaling based on 2000 calorie diet
    portion_size = float((driver.find_element(By.XPATH, '//*[@id="serving-size"]').text)[0:-2])
    
    saturated_fat_ratio = (nutrients.get('Saturated fatty acids') / (portion_size * 22)) * calories #g 
    unsaturated_fat_ratio = (unsaturated_fat / (portion_size * 54)) * calories #g
    trans_fat_ratio = (trans_fat / (portion_size * 2)) * calories #g

    dietary_fiber_ratio = (nutrients.get('Fiber') / (portion_size * 28)) * calories #g
    total_sugars_ratio = (nutrients.get('Sugars') / (portion_size * 50)) * calories #g
    starch_ratio = (starch / (portion_size * 197)) * calories #g

    protein_ratio = (nutrients.get('Protein') / (portion_size * 50)) * calories #g
    cholesterol_ratio = (nutrients.get('Cholesterol') / (portion_size*1000 * 300)) #mg
    sodium_ratio = (nutrients.get('Sodium, Na') / (portion_size*1000 * 2300)) #mg

    vitamin_A_ratio = (nutrients.get('Vitamin A, RAE') / (portion_size*1000000 * 900)) #mcg
    vitamin_B6_ratio = (nutrients.get('Vitamin B6') / (portion_size*1000 * 1.7)) #mg
    vitamin_B12_ratio = (nutrients.get('Vitamin B12') / (portion_size*1000000 * 2.4)) #mcg
    vitamin_C_ratio = (nutrients.get('Vitamin C') / (portion_size*1000 * 90)) #mg
    vitamin_D_ratio = (nutrients.get('Vitamin D') / (portion_size*1000000 * 20)) #mcg
    vitamin_E_ratio = (nutrients.get('Vitamin E (alpha-tocopherol)') / (portion_size*1000 * 15)) #mg
    vitamin_K_ratio = (nutrients.get('Vitamin K') / (portion_size*1000000 * 120)) #mcg
    calcium_ratio = (nutrients.get('Calcium, Ca') / (portion_size*1000 * 1300)) #mg
    copper_ratio = (nutrients.get('Copper, Cu') / (portion_size*1000 * 0.9)) #mg
    iron_ratio = (nutrients.get('Iron, Fe') / (portion_size*1000 * 18)) #mg
    magnesium_ratio = (nutrients.get('Magnesium, Mg') / (portion_size*1000 * 420)) #mg
    phosphorus_ratio = (nutrients.get('Phosphorus, P') / (portion_size*1000 * 1250)) #mg
    potassium_ratio = (nutrients.get('Potassium, K') / (portion_size*1000 * 4700)) #mg
    selenium_ratio = (nutrients.get('Selenium, Se') / (portion_size*1000 * 55)) #mg
    zinc_ratio = (nutrients.get('Zinc, Zn') / (portion_size*1000 * 11)) #mg
    riboflavin_ratio = (nutrients.get('Riboflavin') / (portion_size*1000 * 1.5)) #mg
    niacin_ratio = (nutrients.get('Niacin') / (portion_size*1000 * 16.5)) #mg
    folate_ratio = (nutrients.get('Folate, DFE') / (portion_size*1000000 * 425)) #mcg
    thiamin_ratio = (nutrients.get('Thiamin') / (portion_size*1000 * 1.5)) #mg
    
    # Calculate valuation
    total_vitamins_ratio = (vitamin_A_ratio + vitamin_B6_ratio + vitamin_B12_ratio + vitamin_C_ratio + vitamin_D_ratio + vitamin_E_ratio + vitamin_K_ratio + riboflavin_ratio + niacin_ratio + folate_ratio + thiamin_ratio)
    total_mineral_ratio = (calcium_ratio + copper_ratio + iron_ratio + magnesium_ratio + phosphorus_ratio + potassium_ratio + selenium_ratio + zinc_ratio)
    valuation = (total_vitamins_ratio + total_mineral_ratio + unsaturated_fat_ratio + protein_ratio + dietary_fiber_ratio) - (total_sugars_ratio + saturated_fat_ratio + trans_fat_ratio + starch_ratio + cholesterol_ratio + sodium_ratio)
    
    # Rank given based on valuation
    if valuation > -.5:
        rank = 5
    elif valuation > -1:
        rank = 4
    elif valuation > -2:
        rank = 3
    elif valuation > -3:
        rank = 2
    else:
        rank = 1

    return rank


# This function ties all other functions together
def function_sequence():
    print("\n\n\n**NOTE:**\nSometimes the food names must be exact, or else it will give incorrect results, which you may be able to deduce given the score is weird for the food it is.\nGet exact names directly from https://www.nutritionvalue.org \n")
    food = input("Enter food to get rank of: ")
    print("\n\n\n")
    navigate(food)
    nutrients = get_nutrient_data()
    rank = calculate_rank(nutrients)
    print(f"The rank of {food} is {rank}.")
    driver.close()

function_sequence()