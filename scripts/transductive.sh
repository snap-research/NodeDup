###  GSage --set the "augment" menthod to "none"###
python main.py --datasets=cora --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc
python main.py --datasets=citeseer --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001 --transductive=transduc
python main.py --datasets=coauthor-cs --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001 --transductive=transduc
python main.py --datasets=coauthor-physics --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001 --transductive=transduc
python main.py --datasets=amazon-computers --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc
python main.py --datasets=amazon-photos --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc
python main.py --datasets=igb-tiny --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc

###  NodeDup(L) --set the "augment" menthod to "self_loop"###
python main.py --datasets=cora --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc
python main.py --datasets=citeseer --encoder=sage --predictor=sum --augment=self_loop --augment_times=2 --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001 --transductive=transduc
python main.py --datasets=coauthor-cs --encoder=sage --predictor=sum --augment=self_loop --augment_times=2 --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001 --transductive=transduc
python main.py --datasets=coauthor-physics --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001 --transductive=transduc
python main.py --datasets=amazon-computers --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc
python main.py --datasets=amazon-photos --encoder=sage --predictor=sum --augment=self_loop --augment_times=2 --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc
python main.py --datasets=igb-tiny --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc

###  NodeDup --set the "augment" menthod to "duplicated"###
python main.py --datasets=cora --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc
python main.py --datasets=citeseer --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001 --transductive=transduc
python main.py --datasets=coauthor-cs --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001 --transductive=transduc
python main.py --datasets=coauthor-physics --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001 --transductive=transduc
python main.py --datasets=amazon-computers --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0003 --transductive=transduc
python main.py --datasets=amazon-photos --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc
python main.py --datasets=igb-tiny --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005 --transductive=transduc