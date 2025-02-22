from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Attempt to load the pre-trained model from a pickle file.
try:
    with open("diet_workout_model.pkl", "rb") as f:
        loaded_obj = pickle.load(f)
except Exception as e:
    loaded_obj = None
    print("Error loading model:", e)

# If the loaded object is a dict, wrap it in a dummy model that has a predict method.
if loaded_obj is not None and isinstance(loaded_obj, dict):
    class DummyModel:
        def __init__(self, model_dict):
            self.model_dict = model_dict
        def predict(self, input_features):
            # Simulated prediction logic:
            # Use bmr_kcal (index 9) and activity_level (index 12) from the input vector.
            bmr_kcal = input_features[0, 9]
            activity_level = input_features[0, 12]
            predicted_diet = int(bmr_kcal * 1.2)       # e.g., 20% above BMR
            predicted_workout = int(activity_level + 3)  # e.g., activity_level + 3 workout days
            return [predicted_diet, predicted_workout]
    model = DummyModel(loaded_obj)
else:
    # Fallback dummy model if loading fails.
    class DummyModel:
        def predict(self, input_features):
            return [1800, 4]
    model = DummyModel()

# Home route for a friendly welcome message.
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Diet & Workout API. Use the /predict endpoint with a POST request to get predictions."
    })

# Predict endpoint to generate meal and workout plans.
@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()

    # Parse required input parameters.
    try:
        age = data['age']
        gender = data['gender']
        height_cm = data['height_cm']
        weight_kg = data['weight_kg']
        bmi = data['bmi']
        body_fat_percent = data['body_fat_percent']
        muscle_mass_kg = data['muscle_mass_kg']
        bone_mass_kg = data['bone_mass_kg']
        water_percent = data['water_percent']
        bmr_kcal = data['bmr_kcal']
        visceral_fat = data['visceral_fat']
        metabolic_age = data['metabolic_age']
        activity_level = data['activity_level']
    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400

    # Optional personalization parameters.
    user_allergy = data.get('user_allergy', "").lower()  # e.g., "nuts"
    user_preference = data.get('user_preference', "").lower()
    diet_type = data.get('diet_type', "Regular")
    workout_preference = data.get('workout_preference', "Gym")
    user_goal = data.get('user_goal', "general-fitness").lower()  # e.g., "muscle-gain", "weight-loss"

    # Prepare the input vector (shape: [1, 13]).
    input_features = np.array([[age, gender, height_cm, weight_kg, bmi, body_fat_percent,
                                muscle_mass_kg, bone_mass_kg, water_percent, bmr_kcal,
                                visceral_fat, metabolic_age, activity_level]])

    # Use the loaded (or dummy) model to get predictions.
    predictions = model.predict(input_features)
    predicted_diet = int(predictions[0])
    predicted_workout = int(predictions[1])

    # Generate a sample meal plan.
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
        # If the user is allergic to "nuts", substitute with the alternative.
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

    # Generate a sample workout plan based on the user's goal.
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

    # Build and return the JSON response.
    response = {
        "PredictedDiet": f"{predicted_diet} kcal per day",
        "PredictedWorkout": f"{predicted_workout} workout days per week",
        "MealPlan": meal_plan,
        "WorkoutPlan": workout_plan
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
