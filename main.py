import timeseries as ts
import GeneticAlgorithm as ga
import numpy

sarima1000007 = ts.TimeSeries((1, 0, 0), (0, 0, 0, 7))
sarima0100007 = ts.TimeSeries((0, 1, 0), (0, 0, 0, 7))
sarima0010007 = ts.TimeSeries((0, 0, 1), (0, 0, 0, 7))
sarima0001007 = ts.TimeSeries((0, 0, 0), (1, 0, 0, 7))
sarima0000107 = ts.TimeSeries((0, 0, 0), (0, 1, 0, 7))
sarima0000017 = ts.TimeSeries((0, 0, 0), (0, 0, 1, 7))


prediction = ga.GeneticAlgorithm(sarima1000007.predictions, 
                                sarima0100007.predictions, 
                                sarima0100007.predictions, 
                                sarima0001007.predictions, 
                                sarima0000107.predictions, 
                                sarima0000017.predictions, 
                                sarima1000007.test_data,
                                100)

print(prediction.solution)
