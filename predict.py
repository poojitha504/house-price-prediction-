import pickle

model = pickle.load(open("model.pkl", "rb"))

area = float(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))

result = model.predict([[area, bedrooms, bathrooms]])

print("Predicted Price:", result[0])