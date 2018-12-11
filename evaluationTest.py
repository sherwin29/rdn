import code

from rmllib.data.load import InfluencerMedians
from rmllib.models.conditional import RelationalNaiveBayes
from evaluation import rocCompareWithAlgos

from loadData import AcademicPerformance

if __name__ == '__main__':

    datasets = []
    datasets.append(InfluencerMedians(name='Influencer Medians', sparse=True).node_sample_mask(.7))
    datasets.append(AcademicPerformance(name='Academic Performance', subfeatures=None, sparse=True).node_sample_mask(.7))

    rdnModels = []
    rdnModels.append(RelationalNaiveBayes(name='NB', learn_method='iid', infer_method='iid', calibrate=False))
    rdnModels.append(RelationalNaiveBayes(name='RNB', learn_method='r_iid', infer_method='r_iid', calibrate=False))

    for dataset in datasets:

        rdnPredictionResults = {}
        for rdnModel in rdnModels:
            print("start Evaluation of " + rdnModel.name  + 'rdnModel on ' + dataset.name + ' dataset')
            train_data = dataset.create_training().copy()
            rdnModel.fit(train_data)
            rdnModel.predictions = rdnModel.predict_proba(train_data)

            rdnPredictionResults[rdnModel.name] = rdnModel.predictions

        rocCompareWithAlgos(dataset, rdnPredictionResults)

    