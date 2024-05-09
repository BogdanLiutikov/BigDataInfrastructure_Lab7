from kmean import KMean

kmean = KMean()

data = kmean.load_data('data/en.openfoodfacts.org.products.csv', 1_000)


feature_data = kmean.make_feature_column(data)
scaled_data = kmean.standard_scale(feature_data)
kmean.train(scaled_data)

pred = kmean.predict(scaled_data)
print('KMean Centers\n', kmean.model.clusterCenters())

kmean.plot(pred)

eval = kmean.eval(pred)
print('EVAL', eval)