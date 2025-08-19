import cProfile, pstats, io
from main import experiment  # your entry point
from data.datasets import get_pd_dataset, prepare_and_split_data
from classifiers.TM_log_reg import TwoModelLogReg
data = get_pd_dataset(name="BreastCancer")

    #preprocess and split the dataset into train an test data
X_train, y_train, X_test, y_test = prepare_and_split_data(data = data,
                                                        test_size=0.2,
                                                        c=0.2,
                                                        labeling_mechanism="SCAR",
                                                        train_label_distribution=None,
                                                        test_label_distribution=None,
                                                        scale_data="standard")

clf = TwoModelLogReg(epochs=100, learning_rate=0.001,penalty="l2",solver="adam") 
pr = cProfile.Profile()
pr.enable()
try:
    clf.fit(X_train.values, y_train.values)
except Exception as e:
    print("Error occurred:", e)
finally:
    pr.disable()
print("Profiling results:")

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")  # or "tottime"
ps.print_stats(40)  # top 40
print(s.getvalue())
ps.dump_stats("profile_output.out")  # save to file if needed
