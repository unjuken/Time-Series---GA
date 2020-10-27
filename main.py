import timeseries as ts
import GeneticAlgorithm as ga
import numpy

sarima1000007 = ts.TimeSeries((1, 0, 0), (0, 0, 0, 7))
sarima0100007 = ts.TimeSeries((0, 1, 0), (0, 0, 0, 7))
sarima0010007 = ts.TimeSeries((0, 0, 1), (0, 0, 0, 7))
sarima0001007 = ts.TimeSeries((0, 0, 0), (1, 0, 0, 7))
sarima0000107 = ts.TimeSeries((0, 0, 0), (0, 1, 0, 7))
sarima0000017 = ts.TimeSeries((0, 0, 0), (0, 0, 1, 7))
test_data = sarima1000007.test_data


models = numpy.matrix([
                        sarima1000007.predictions, 
                        sarima0100007.predictions, 
                        sarima0010007.predictions, 
                        sarima0001007.predictions, 
                        sarima0000107.predictions, 
                        sarima0000017.predictions
                        ]).transpose()

models = numpy.where(models < 0, 0, models)


prediction = ga.GeneticAlgorithm(models,
                                test_data,
                                1000)


weights = numpy.matrix(prediction.solution)
GATimeSeries = (models*weights.T).flatten()
npTestData = numpy.matrix(test_data.array)
residuals = npTestData - GATimeSeries
MAPE = round(numpy.mean(abs(residuals/npTestData)),4)
RMSE = numpy.sqrt(numpy.mean(numpy.square(residuals)))


print("(1, 0, 0), (0, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima1000007.RMSE, mape=sarima1000007.MAPE))
print("(0, 1, 0), (0, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0100007.RMSE, mape=sarima0100007.MAPE))
print("(0, 0, 1), (0, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0010007.RMSE, mape=sarima0010007.MAPE))
print("(0, 0, 0), (1, 0, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0001007.RMSE, mape=sarima0001007.MAPE))
print("(0, 0, 0), (0, 1, 0, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0000107.RMSE, mape=sarima0000107.MAPE))
print("(0, 0, 0), (0, 0, 1, 7): RMSE = {rmse}, MAPE = {mape}".format(rmse=sarima0000017.RMSE, mape=sarima0000017.MAPE))
print("GA: RMSE = {rmse}, MAPE = {mape}".format(rmse=RMSE, mape=MAPE))
