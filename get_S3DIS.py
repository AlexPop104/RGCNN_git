from torch_geometric.datasets import S3DIS

root = "/media/rambo/ssd2/Alex_data/RGCNN/S3DIS/"
dataset_train = S3DIS(root=root, test_area=6, train=True)
dataset_test =  S3DIS(root=root, test_area=6, train=False)
