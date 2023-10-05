###  GSage --set the "augment" menthod to "none"###
python main_inductive.py --datasets=cora --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005
python main_inductive.py --datasets=citeseer --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001
python main_inductive.py --datasets=coauthor-cs --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001
python main_inductive.py --datasets=coauthor-physics --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001
python main_inductive.py --datasets=amazon-computers --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005
python main_inductive.py --datasets=amazon-photos --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005
python main_inductive.py --datasets=igb-tiny --encoder=sage --predictor=sum --augment=none --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005

###  NodeDup(L) --set the "augment" menthod to "self_loop"###
python main_inductive.py --datasets=cora --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005
python main_inductive.py --datasets=citeseer --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001
python main_inductive.py --datasets=coauthor-cs --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001
python main_inductive.py --datasets=coauthor-physics --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001
python main_inductive.py --datasets=amazon-computers --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005
python main_inductive.py --datasets=amazon-photos --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005
python main_inductive.py --datasets=igb-tiny --encoder=sage --predictor=sum --augment=self_loop --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005

###  NodeDup --set the "augment" menthod to "duplicated"###
python main_inductive.py --datasets=cora --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005
python main_inductive.py --datasets=citeseer --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001
python main_inductive.py --datasets=coauthor-cs --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001
python main_inductive.py --datasets=coauthor-physics --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0001
python main_inductive.py --datasets=amazon-computers --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005
python main_inductive.py --datasets=amazon-photos --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005
python main_inductive.py --datasets=igb-tiny --encoder=sage --predictor=sum --augment=duplicated --negative_samples=500 --patience=100 --use_valedges_as_input --dropout=0.5 --metric=hits@20 --runs=10 --lr=0.0005