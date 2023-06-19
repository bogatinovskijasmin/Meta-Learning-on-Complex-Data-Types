def basic_hive_cote (data_dir, dataset_name):
    hc = HIVE_COTE()
    train_x , train_y = load_data(data_dir + dataset_name + "_TRAIN.ts")
    hc.fit(train_x, train_y)
    test_x, test_y = load_data(data_dir + dataset_name + "_TEST.ts")
    preds = hc.predict(train_x)
    