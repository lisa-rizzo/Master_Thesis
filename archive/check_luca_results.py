import pickle

file_path = "/home/student/lisa_ma/glare_clean/results/test_glare_07052025_2326/iteration0/grad_norms.pkl"
with open(file_path, "rb") as f:
    luca_results = pickle.load(f)
print(luca_results)