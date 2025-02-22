from flask import Flask, request, jsonify
import pickle
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load the saved model from the pickle file
with open("diet_workout_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

# Rebuild the model from its JSON structure and load weights
model = keras.models.model_from_json(model_dict["model_json"])
model.set_weights(model_dict["model_weights"])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # --- Parse required input parameters for model prediction ---
    try:
        age = data['age']                        # int
        gender = data['gender']                  # int (0: Male, 1: Female)
        height_cm = data['height_cm']            # int
        weight_kg = data['weight_kg']            # int
        bmi = data['bmi']                        # float
        body_fat_percent = data['body_fat_percent']  # float
        muscle_mass_kg = data['muscle_mass_kg']    # float
        bone_mass_kg = data['bone_mass_kg']        # float
        water_percent = data['water_percent']      # float
        bmr_kcal = data['bmr_kcal']              # int
        visceral_fat = data['visceral_fat']        # int
        metabolic_age = data['metabolic_age']      # int
        activity_level = data['activity_level']    # int (0: Sedentary, 1: Moderate, 2: Active)
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400

    # --- Additional parameters for personalization ---
    user_allergy = data.get('user_allergy', "").lower()             # string, e.g., "nuts"
    user_preference = data.get('user_preference', "").lower()           # string, e.g., "high-protein"
    diet_type = data.get('diet_type', "Regular")                        # string, e.g., "Vegan", "Keto", etc.
    workout_preference = data.get('workout_preference', "Gym")          # string, e.g., "Gym", "Outdoor", "Home"
    user_goal = data.get('user_goal', "general-fitness").lower()        # string, e.g., "muscle-gain", "weight-loss"

    # --- Prepare the input vector for the model (shape: [1, 13]) ---
    input_features = np.array([[age, gender, height_cm, weight_kg, bmi, body_fat_percent,
                                muscle_mass_kg, bone_mass_kg, water_percent, bmr_kcal,
                                visceral_fat, metabolic_age, activity_level]])
    
    # --- Run prediction ---
    predicted_diet, predicted_workout = model.predict(input_features)
    predicted_diet = int(predicted_diet[0][0])
    predicted_workout = int(predicted_workout[0][0])

    # --- Generate a sample meal plan ---
    # (If the user is allergic to "nuts", substitute with an alternative food.)
    meals = [
        {"Meal": "Breakfast", "Food": "Oatmeal + Nuts", "Calories": "350 kcal",
         "Alternative": "Whole Wheat Toast", "Allergen": "nuts"},
        {"Meal": "Lunch", "Food": "Grilled Chicken + Peanut Sauce", "Calories": "600 kcal",
         "Alternative": "Tofu + Salad", "Allergen": "nuts"},
        {"Meal": "Dinner", "Food": "Fish + Almond Quinoa", "Calories": "500 kcal",
         "Alternative": "Lentil Soup + Rice", "Allergen": "nuts"}
    ]
    meal_plan = []
    for m in meals:
        # Check if the food contains an allergen (here, "nuts")
        if user_allergy and user_allergy in m["Allergen"].lower():
            meal_plan.append({
                "Meal": m["Meal"],
                "Food": m["Alternative"],
                "Calories": m["Calories"],
                "AllergySafe": True
            })
        else:
            meal_plan.append({
                "Meal": m["Meal"],
                "Food": m["Food"],
                "Calories": m["Calories"],
                "AllergySafe": True
            })

    # --- Generate a sample workout plan ---
    # Only include Strength workouts if the goal is "muscle-gain",
    # only include Cardio if "weight-loss", otherwise include all.
    workouts = [
        {"Exercise": "Bench Press", "Type": "Strength", "Reps/Sets": "4 sets x 8 reps", "CaloriesBurned": "250 kcal"},
        {"Exercise": "Deadlifts", "Type": "Strength", "Reps/Sets": "4 sets x 6 reps", "CaloriesBurned": "300 kcal"},
        {"Exercise": "Cycling", "Type": "Cardio", "Reps/Sets": "30 mins", "CaloriesBurned": "400 kcal"},
        {"Exercise": "Jump Rope", "Type": "Cardio", "Reps/Sets": "15 mins", "CaloriesBurned": "150 kcal"}
    ]
    workout_plan = []
    for w in workouts:
        if user_goal == "muscle-gain" and w["Type"].lower() == "strength":
            workout_plan.append(w)
        elif user_goal == "weight-loss" and w["Type"].lower() == "cardio":
            workout_plan.append(w)
        elif user_goal == "general-fitness":
            workout_plan.append(w)

    # --- Compose the JSON response ---
    response = {
        "PredictedDiet": f"{predicted_diet} kcal per day",
        "PredictedWorkout": f"{predicted_workout} workout days per week",
        "MealPlan": meal_plan,
        "WorkoutPlan": workout_plan
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
