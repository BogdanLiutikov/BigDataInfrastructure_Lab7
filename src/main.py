from .kmean import KMean
from .database import Database

kmean = KMean()

data = kmean.load_data('../../Lab5/Code/data/en.openfoodfacts.org.products.csv', 100)


feature_data = kmean.make_feature_column(data)
scaled_data = kmean.standard_scale(feature_data)
kmean.train(scaled_data)


db = Database()

new_prediction = [{'energy-kcal_100g': 150., 'fat_100g': 15.,
                  'proteins_100g': 10., 'carbohydrates_100g': 50.},
                  {'energy-kcal_100g': 55., 'fat_100g': 15.,
                  'proteins_100g': 55., 'carbohydrates_100g': 13.}]

dataframe = kmean.make_feature_column(kmean.get_dataframe_fromdict(new_prediction))

new_prediction = kmean.predict(dataframe, scale=True)

new_prediction = new_prediction.select('features', 'prediction').toPandas().to_dict(orient='records')

print(new_prediction)
s = next(db.get_session())
db.create_record(s, new_prediction)